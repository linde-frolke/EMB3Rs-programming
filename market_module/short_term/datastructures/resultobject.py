import cvxpy as cp
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
import itertools
from datetime import datetime
import pickle

from ...short_term.datastructures.inputstructs import AgentData, MarketSettings
from ...short_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder
from ...short_term.plotting_processing_functions.plot_pool_clearing import prep_plot_market_clearing_pool
from ...short_term.plotting_processing_functions.bool_to_string import bool_to_string


class ResultData:
    def __init__(self, prob_status, day_nrs,
                 Pn_t, Ln_t, Gn_t, shadow_price_t,
                 agent_data: AgentData, settings: MarketSettings,
                 Tnm_t=None, Snm_t=None, Bnm_t=None, 
                 q_comm_t=None, q_exp_t=None, q_imp_t=None):
        """
        Object to store relevant outputs from a solved market problem.
        Initialization only extracts necessary values from the optimization
        Functions can be used to compute other quantities or generate plots

        :param prob: cvxpy problem object
        :param cb: ConstraintBuilder used for cvxpy problem
        :param agent_data: AgentData object
        :param settings: MarketSettings object
        """
        #
        self.market = settings.market_design

        if prob_status == False:
            self.optimal = False
            raise RuntimeError(
                "problem is not solved to an optimal solution. days with nrs " + str(day_nrs) +
                "were not solved to optimality")
        else:
            self.optimal = True
            # store values of the optimized variables -------------------------------------------------------
            self.Pn = Pn_t
            self.Ln = Ln_t
            self.Gn = Gn_t

            if settings.market_design == "p2p":
                # extract trade variable - a square dataframe for each time index
                self.Tnm = Tnm_t
                self.Bnm = Bnm_t
                self.Snm = Snm_t
            elif settings.market_design == "community":
                self.Tnm = Tnm_t
                self.qimp = q_imp_t
                self.qexp = q_exp_t
                self.qcomm = q_comm_t
                
            else:
                self.Tnm = "none"

            # get values related to duals  ----------------------------------------
            self.shadow_price = shadow_price_t

            # initialize empty slots for uncomputed result quantities
            self.QoE = None
            self.social_welfare_h = None
            self.settlement = None
            # fill the empty slots
            self.compute_output_quantities(settings, agent_data)

    # a function to make all relevant output variables
    def compute_output_quantities(self, settings, agent_data):
        # get shadow price, Qoe, for different markets --------------------------------------------------
        if settings.market_design == "p2p":
            # QoE
            self.QoE = pd.DataFrame(index=range(settings.nr_of_h), columns=["QoE"])
            for t in range(0, settings.nr_of_h):
                lambda_j = []
                for a1 in agent_data.agent_name:
                    for a2 in agent_data.agent_name:
                        if self.Pn[a1][t] != 0:  # avoid #DIV/0! error
                            lambda_j.append(
                                agent_data.cost[a1][t] * self.Tnm[t][a1][a2] / self.Pn[a1][t])
                        if self.Ln[a1][t] != 0:  # avoid #DIV/0! error
                            lambda_j.append(
                                agent_data.util[a1][t] * self.Tnm[t][a1][a2] / self.Ln[a1][t])
                if len(lambda_j) == 0:  # If no power is traded in t
                    self.QoE["QoE"][t] = 'Not Defined'
                elif (max(lambda_j) - min(lambda_j)) != 0:  # avoid #DIV/0! error
                    self.QoE["QoE"][t] = (
                        1 - (st.pstdev(lambda_j) / (max(lambda_j) - min(lambda_j))))
                else:
                    pass
            
        else:
            self.QoE = pd.DataFrame({"QoE" : "not applicable"}, index=[0])

        # hourly social welfare an array of length settings.nr_of_h, same for all markets
        self.social_welfare_h = pd.DataFrame(index=range(settings.nr_of_h), columns=["Social Welfare"])
        for t in range(0, settings.nr_of_h):
            total_cost = np.sum(np.multiply(agent_data.cost.T[t], self.Gn.T[t]))
            total_util = np.sum(np.multiply(agent_data.util.T[t], self.Ln.T[t]))
            self.social_welfare_h["Social Welfare"][t] = (total_cost - total_util)

        # Settlement has standard format:
        self.settlement = pd.DataFrame(index=range(settings.nr_of_h), columns=agent_data.agent_name)
        if settings.market_design == "p2p":
            for t in range(0, settings.nr_of_h):
                for agent in agent_data.agent_name:
                    self.settlement.loc[t, agent] = sum([self.shadow_price[t].loc[agent, agent2] * self.Snm[t].loc[agent, agent2] -
                                                         self.shadow_price[t].loc[agent2, agent] * self.Bnm[t].loc[agent, agent2]
                                                        for agent2 in agent_data.agent_name])

        elif settings.market_design == "pool":
            if settings.network_type is None:
                for t in range(0, settings.nr_of_h):
                    for agent in agent_data.agent_name:
                        self.settlement[agent][t] = self.shadow_price['uniform price'][t]*(self.Gn[agent][t] -
                                                                                           self.Ln[agent][t])
            elif settings.network_type == "direction":
                for t in range(settings.nr_of_h):
                    for agent in agent_data.agent_name:
                        self.settlement[agent][t] = self.shadow_price[agent][t] * \
                            self.Pn[agent][t] # TODO check this
            elif settings.network_type == "size":
                raise NotImplementedError(
                    "network-aware size is not implemented yet")
        elif settings.market_design == "community":
            for agent in range(agent_data.nr_of_agents):
                if agent_data.is_in_community[agent]:
                    self.settlement.iloc[:,agent] = (self.shadow_price["community"].values * self.qcomm.iloc[:,agent].values + 
                            self.shadow_price["export"].values * self.qexp.iloc[:,agent].values -
                            self.shadow_price["import"].values * self.qimp.iloc[:,agent].values  
                        )
                else:
                    self.settlement.iloc[:,agent] = self.shadow_price["non-community"].values * self.Pn.iloc[:,agent].values
            if (self.settlement.sum(axis=1) > 10**-7).any():
                raise Warning("There is a mistake in the computation of the setttlement. At any time, the sum of settlement over all agents must be nonpositive. ")


    # a function working on the result object, to create plots
    def plot_market_clearing(self, period: int, settings: MarketSettings, agent_data: AgentData, outfile: str):
        """
        makes a plot in the file named at path outfile.
        """
        if not period <= settings.nr_of_h:
            raise ValueError("period should be in range(settings.nr_of_h)")

        if self.market == "pool" and settings.network_type is None:
            data1, data2, yy_gen, yy_load = prep_plot_market_clearing_pool(
                period, agent_data)
            # Plotting
            plt.step(data1, yy_gen)  # Generation curve
            plt.step(data2, yy_load)  # Load curve
            # (heat negotiated,shadow price)
            plt.plot(np.sum(self.Ln.T[period]), abs(
                self.shadow_price.T[period]), 'ro')
            plt.ylabel('Price (€/kWh)')
            plt.xlabel('Heat (kWh)')
            plt.savefig(fname=outfile)
            return "success"
        else:
            return print("not implemented yet")

    def save_as_pickle(self, path_to_file=None):
        if path_to_file is None:
            # generate file path
            today = datetime.now()
            filename = self.market + \
                today.strftime("%d%m%Y_%H:%M:%S") + ".pickle"
            path_to_file = "./pickled_data/" + filename

        # open a file, where you want to store the data
        file = open(path_to_file, 'wb')
        # dump information to that file
        pickle.dump(self, file)
        # close the file
        file.close()

    def convert_to_dicts(self):
        print("converting result to dictionaries")
        return_dict = {'Gn': self.Gn.to_dict(orient="list"),
                       'Ln': self.Ln.to_dict(orient="list"),
                       'Pn': self.Pn.to_dict(orient="list"),
                       'QoE': self.QoE.to_dict(orient="list")["QoE"],
                       'optimal': bool_to_string(self.optimal),
                       'settlement' : self.settlement.to_dict(orient="list"),
                       'social_welfare_h': self.social_welfare_h.to_dict(orient="list")#,
                       #"agent_operational_cost" : self.agent_op_cost.to_dict(orient="list")
                       }
        if self.market == "p2p":
            return_dict['Tnm'] = [self.Tnm[t].to_dict(orient="list") for t in range(len(self.Tnm))]
            return_dict['shadow_price'] = [self.shadow_price[t].to_dict(orient="list") 
                                            for t in range(len(self.shadow_price))]
        
        elif self.market == "community":
            return_dict['Tnm'] = self.Tnm.to_dict(orient="list")
            return_dict['shadow_price'] = abs(self.shadow_price).to_dict(orient="list")
            return_dict['q_imp'] = self.qimp.to_dict(orient="list")
            return_dict['q_exp'] = self.qexp.to_dict(orient="list")
            return_dict['q_comm'] = self.qcomm.to_dict(orient="list")
        else:
            return_dict['Tnm'] = self.Tnm
            return_dict['shadow_price'] = abs(self.shadow_price).to_dict(orient="list")

        return return_dict
