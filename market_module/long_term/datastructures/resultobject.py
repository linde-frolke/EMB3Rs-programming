import cvxpy as cp
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings
from ...long_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder
from ...short_term.plotting_processing_functions.bool_to_string import bool_to_string
import itertools


class ResultData:
    def __init__(self, prob: cp.problems.problem.Problem,
                 cb: ConstraintBuilder,
                 agent_data: AgentData, settings: MarketSettings, Pn_t, Ln_t, Gn_t, shadow_price_t, Tnm_t=None, 
                 En=None, Bn=None):
        """
        Object to store relevant outputs from a solved market problem.
        Initialization only extracts necessary values from the optimization
        Functions can be used to compute other quantities or generate plots

        :param name: str
        :param prob: cvxpy problem object
        :param cb: ConstraintBuilder used for cvxpy problem
        :param agent_data: AgentData object
        :param settings: MarketSettings object
        """
        #
        #self.name = name

        self.market = settings.market_design

        if prob.status in ["infeasible", "unbounded"]:
            self.optimal = False
            raise Warning(
                "problem is not solved to an optimal solution. result object will not contain any info")
        else:
            self.optimal = True
            # store values of the optimized variables -------------------------------------------------------
            self.storage_present = (agent_data.nr_of_stor > 0)
            self.Pn = Pn_t
            self.Ln = Ln_t
            self.Gn = Gn_t
            self.En = En
            self.Bn = Bn

            if settings.market_design == "decentralized":
                # extract trade variable - a square dataframe for each time index
                self.Tnm = Tnm_t
            else:
                self.Tnm = None

            # get values related to duals  ----------------------------------------
            if settings.market_design == "centralized":
                #self.shadow_price = cb.get_constraint(
                    #str_="powerbalance").dual_value
                self.shadow_price = pd.DataFrame(
                    shadow_price_t, columns=["uniform price"])

            elif settings.market_design == "decentralized":

                self.shadow_price = shadow_price_t

            # initialize empty slots for uncomputed result quantities
            self.ADG = None
            self.expensive_prod = None
            self.QoE = None
            self.social_welfare_h = None
            self.settlement = None
            self.agent_operational_cost = None
            self.SPM = None
            # compute outputs
            self.compute_output_quantities(settings, agent_data)
            if agent_data.fbp_agent is not None:
                self.best_price = self.find_best_price(agent_data.fbp_time, agent_data.fbp_agent, agent_data, settings)

    # a function to make all relevant output variables
    def compute_output_quantities(self, settings, agent_data):
        # get shadow price, Qoe, for different markets --------------------------------------------------
        if settings.market_design == "centralized":
            self.QoE = pd.DataFrame(index=range(
                settings.diff), columns=["QoE"])
            for t in range(0, settings.diff):
                self.QoE["QoE"][t] = 'Not Defined'

            # raise Warning("QoE not implemented for pool")
        elif settings.market_design == "decentralized":
            # QoE
            self.QoE = pd.DataFrame(index=range(
                settings.diff), columns=["QoE"])

            for t in range(0, settings.diff):
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
            # self.qoe = np.average(self.QoE) # we only need it for each hour.

        # hourly social welfare an array of length agent_data.day_range*settings.recurrence*agent_data.data_size
        self.social_welfare_h = pd.DataFrame(index=range(
            settings.diff), columns=["Social Welfare"])
        for t in range(0, settings.diff):
            total_cost = np.sum(np.multiply(
                agent_data.cost.T[t], self.Gn.T[t]))
            total_util = np.sum(np.multiply(
                agent_data.util.T[t], self.Ln.T[t]))
            self.social_welfare_h["Social Welfare"][t] = (
                    total_cost - total_util)

        # Settlement
        self.settlement = pd.DataFrame(index=range(
            settings.diff), columns=agent_data.agent_name)
        if settings.market_design == "decentralized":
            for t in range(0, settings.diff):
                for agent in agent_data.agent_name:
                    aux = []
                    for agent2 in agent_data.agent_name:
                        aux.append(self.shadow_price[t].loc[agent, agent2] * self.Tnm[t].loc[agent, agent2]
                                   )
                    self.settlement[agent][t] = sum(aux)
        elif settings.market_design == "centralized":
            for t in range(0, settings.diff):
                for agent in agent_data.agent_name:
                    self.settlement[agent][t] = self.shadow_price['uniform price'][t] * (self.Gn[agent][t] -
                                                                                         self.Ln[agent][t])

            if self.En is not None:
                # add storage settlement to the settlement output
                storage_settlement = pd.DataFrame(index=range(settings.diff), columns=agent_data.storage_name)
                for t in range(0, settings.diff):
                    for stor in agent_data.storage_name:
                        storage_settlement[stor][t] = self.shadow_price['uniform price'][t] * - self.Bn[stor][t]
                
                self.settlement = pd.concat([self.settlement, storage_settlement], axis=1)

                # add storage net despatch to Pn output (using that Pn = -Bn for the storage)
                storage_Pn = pd.DataFrame(index=range(settings.diff), columns=agent_data.storage_name)
                storage_Pn = - self.Bn.iloc[:,:]
                self.Pn = pd.concat([self.Pn, storage_Pn], axis=1)

                # storage_operational_cost (always zero)
                storage_operational_cost = pd.DataFrame(0, index=range(settings.diff), columns=agent_data.storage_name)
                

        # agent_operational_cost
        self.agent_operational_cost = pd.DataFrame(index=range(
            settings.diff), columns=agent_data.agent_name)
        for t in range(0, settings.diff):
            for agent in agent_data.agent_name:
                    self.agent_operational_cost[agent][t] = agent_data.cost[agent][t] * self.Gn[agent][t]
        if self.En is not None: # add storage operational cost (always zero, this is for use in BM)
            self.agent_operational_cost = pd.concat([self.agent_operational_cost, storage_operational_cost], axis=1)

        # list with producers+prosumers
        prod_pros = []
        for agent in agent_data.agent_name:
            aux = []
            for t in range(0, settings.diff):
                aux.append(agent_data.gmax[agent][t] - agent_data.lmax[agent][t])
            if max(aux) > 0 and agent not in prod_pros:
                prod_pros.append(agent)

        # AVERAGE DISPATCHED GENERATION (ADG)
        self.ADG = pd.DataFrame(index=['ADG'], columns=prod_pros)
        for agent in prod_pros:
            aux = []
            for t in range(0, settings.diff):
                if agent_data.gmax[agent][t] == 0:
                    aux.append(0)  # if source production is zero
                else:
                    aux.append(self.Gn[agent][t] / agent_data.gmax[agent][t])
            self.ADG[agent]['ADG'] = np.average(aux) * 100
        # SUCCESSFUL PARTICIPATION IN THE MARKET (SPM)
        self.SPM = pd.DataFrame(index=['SPM'], columns=prod_pros)
        for agent in prod_pros:
            aux = []
            for t in range(0, settings.diff):
                if agent_data.gmax[agent][t] == 0:
                    aux.append(0)
                else:
                    # check small numbers (ex: 1E-15)
                    if self.Gn[agent][t] != 0:
                        aux.append(1)
                    else:
                        aux.append(0)
            self.SPM[agent]['SPM'] = np.average(aux) * 100

    def find_best_price(self, period: int, agent_name, agent_data, settings):
        if settings.market_design == "decentralized":
            raise ValueError(
                "Find the best price option is only available in centralized market")
        elif period >= settings.diff:
            raise ValueError(
                'Please select a period within the simulation time range')
        else:
            aux = []
            for name in agent_data.agent_name:
                if self.Gn[name][period] > 0:
                    aux.append([agent_data.cost[name][period], name])
            if len(aux) == 0:
                print('No production dispatched')
            else:
                self.expensive_prod = max(aux, key=lambda x: x[0])
                return agent_data.cost[self.expensive_prod[1]][period]

    def convert_to_dicts(self):
        return_dict = {'Gn': self.Gn.to_dict(orient='list'),
                       'Ln': self.Ln.to_dict(orient='list'),
                       'Pn': self.Pn.to_dict(orient='list'),
                       'optimal': bool_to_string(self.optimal),
                       'settlement': self.settlement.to_dict(orient='list'),
                       'agent_operational_cost': self.agent_operational_cost.to_dict(orient='list'),
                       'social_welfare_h': self.social_welfare_h.values.T.tolist()[0],
                       'SPM': self.SPM.transpose().to_dict()['SPM'],
                       'ADG': self.ADG.transpose().to_dict()['ADG'],
                       'expensive_prod': self.expensive_prod,
                       'QoE': self.QoE.to_dict()['QoE']
                       }
        if self.market == 'centralized':
            return_dict['shadow_price'] = abs(self.shadow_price).to_dict(orient='list')['uniform price']
            return_dict['Tnm'] = "none"
            if self.storage_present:
                return_dict["En"] = self.En.to_dict(orient="list")
                return_dict["Bn"] = self.Bn.to_dict(orient="list")
            else:
                return_dict["En"] = "none"
                return_dict["Bn"] = "none"    
        else:
            return_dict['shadow_price'] = [abs(self.shadow_price[t]).to_dict(orient="list")
                                           for t in range(len(self.shadow_price))]
            return_dict['Tnm'] = [self.Tnm[t].to_dict(orient="list") for t in range(len(self.Tnm))]
            return_dict["En"] = "none"
            return_dict["Bn"] = "none"
        try:
            return_dict['best_price'] = self.best_price
        except:
            pass
            
        return return_dict
