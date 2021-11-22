import cvxpy as cp
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from datastructures.inputstructs import AgentData, MarketSettings
from constraintbuilder.ConstraintBuilder import ConstraintBuilder
from plotting_processing_functions.plot_pool_clearing import prep_plot_market_clearing_pool
import itertools
from datetime import datetime
import pickle

class ResultData:
    def __init__(self, name, prob: cp.problems.problem.Problem,
                 cb: ConstraintBuilder,
                 agent_data: AgentData, settings: MarketSettings,
                 network_data=None):
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
        self.name = name
        self.market = settings.market_design

        if prob.status in ["infeasible", "unbounded"]:
            self.optimal = False
            raise Warning("problem is not solved to an optimal solution. result object will not contain any info")
        else:
            self.optimal = True
            # store values of the optimized variables -------------------------------------------------------
            variables = prob.variables()
            varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]
            self.Pn = pd.DataFrame(variables[varnames.index("Pn")].value, columns=agent_data.agent_name)
            self.Ln = pd.DataFrame(variables[varnames.index("Ln")].value, columns=agent_data.agent_name)
            self.Gn = pd.DataFrame(variables[varnames.index("Gn")].value, columns=agent_data.agent_name)
            if settings.market_design == "p2p":
                # extract trade variable - a square dataframe for each time index
                self.Tnm = [pd.DataFrame(variables[varnames.index("Tnm_" + str(t))].value,
                                         columns=agent_data.agent_name, index=agent_data.agent_name)
                            for t in range(settings.nr_of_h)]

            elif settings.market_design == "community":
                trade_array = np.column_stack([variables[varnames.index("q_imp")].value,
                                               variables[varnames.index("q_exp")].value])
                self.Tnm = pd.DataFrame(trade_array, index=settings.timestamps, columns=["q_imp", "q_exp"])

            # get values related to duals  ----------------------------------------
            if settings.market_design == "pool":
                if settings.offer_type == 'block':
                    self.shadow_price = []
                    for t in settings.timestamps: 
                        max_cost_disp = []
                        for agent in agent_data.agent_name:
                            if self.Gn[agent][t] > 0:
                                max_cost_disp.append(agent_data.cost[agent][t])        
                        if len(max_cost_disp) > 0:
                            self.shadow_price.append(max(max_cost_disp))
                        else:  # if there is no generation
                            self.shadow_price.append(min(agent_data.cost.T[t]))
                    self.shadow_price = pd.DataFrame(self.shadow_price, columns=["uniform price"])
                elif settings.network_type is not None:
                    if settings.network_type == "direction":  # in this case we have nodal prices
                        nodal_prices = np.zeros((settings.nr_of_h, network_data.nr_of_n))
                        for t in settings.timestamps:
                            for n in range(network_data.nr_of_n):
                                nodal_prices[t, n] = cb.get_constraint(str_="def_nodal_P" + str(t) + "_" +
                                                                            str(network_data.N[n])).dual_value
                        # shadow_price = np.array((settings.nr_of_h, agent_data.nr_of_agents))
                        shadow_price = nodal_prices[:, network_data.loc_a]
                        self.shadow_price = pd.DataFrame(shadow_price, columns=agent_data.agent_name)
                    if settings.network_type == "size":  # in this case we have nodal prices
                        raise NotImplementedError("this is to be implemented")
                # if no special case of pool market, the output is simple:
                else:
                    self.shadow_price = cb.get_constraint(str_="powerbalance").dual_value
                    self.shadow_price = pd.DataFrame(self.shadow_price, columns=["uniform price"])
            # p2p market
            elif settings.market_design == "p2p":
                self.shadow_price = [pd.DataFrame(index=agent_data.agent_name, columns=agent_data.agent_name)
                                     for t in settings.timestamps]
                for t in settings.timestamps:
                    if settings.offer_type == 'block':
                        if settings.product_diff == 'noPref':
                            max_cost_disp=[]
                            for agent in agent_data.agent_name:
                                if self.Gn[agent][t] > 0:
                                    max_cost_disp.append(agent_data.cost[agent][t])
                            for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
                                if len(max_cost_disp) > 0:
                                    self.shadow_price[t].iloc[i, j] = max(max_cost_disp)
                                elif len(max_cost_disp) == 0: # if there is no generation
                                    self.shadow_price[t].iloc[i, j] = min(agent_data.cost.T[t])
                                    
                                if j == i:
                                    self.shadow_price[t].iloc[i, j] = 0
                        else:
                            for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
                                self.shadow_price[t].iloc[i, j] = agent_data.cost[agent_data.agent_name[i]][t]
                                if j == i:
                                    self.shadow_price[t].iloc[i, j] = 0

                    else:
                        for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
                            # if not i == j:
                            if j >= i:
                                constr_name = "reciprocity_t" + str(t) + str(i) + str(j)
                                self.shadow_price[t].iloc[i, j] = cb.get_constraint(str_=constr_name).dual_value
                                self.shadow_price[t].iloc[j, i] = - self.shadow_price[t].iloc[i, j]

            elif settings.market_design == "community":
                price_array = np.column_stack([cb.get_constraint(str_="internal_trades").dual_value,
                                              cb.get_constraint(str_="total_exp").dual_value,
                                              cb.get_constraint(str_="total_imp").dual_value,
                                              cb.get_constraint(str_="noncom_powerbalance").dual_value])
                self.shadow_price = pd.DataFrame(price_array, index=settings.timestamps,
                                                 columns=["community", "export", "import", "non-community"])

            # initialize empty slots for uncomputed result quantities
            self.QoE = None
            self.social_welfare_h = None
            self.settlement = None
            # fill the empty slots
            self.compute_output_quantities(settings, agent_data)

    # a function to make all relevant output variables
    def compute_output_quantities(self, settings, agent_data):
        # get shadow price, Qoe, for different markets --------------------------------------------------
        if settings.market_design == "pool":
            self.QoE = np.nan * np.ones(settings.nr_of_h)
            # raise Warning("QoE not implemented for pool")
        elif settings.market_design == "p2p":
            # QoE
            self.QoE = pd.DataFrame(index=range(settings.nr_of_h), columns=["QoE"])
            for t in range(0, settings.nr_of_h):
                self.lambda_j = []
                for a1 in agent_data.agent_name:
                    for a2 in agent_data.agent_name:
                        if self.Pn[a1][t] != 0:  # avoid #DIV/0! error
                            self.lambda_j.append(agent_data.cost[a1][t] * self.Tnm[t][a1][a2] / self.Pn[a1][t])
                        if self.Ln[a1][t] != 0:  # avoid #DIV/0! error
                            self.lambda_j.append(agent_data.util[a1][t] * self.Tnm[t][a1][a2] / self.Ln[a1][t])
                if len(self.lambda_j) == 0:  # If no power is traded in t
                    self.QoE["QoE"][t] = 'Not Defined'
                elif (max(self.lambda_j) - min(self.lambda_j)) != 0:  # avoid #DIV/0! error
                    self.QoE["QoE"][t] = (1 - (st.pstdev(self.lambda_j) / (max(self.lambda_j) - min(self.lambda_j))))
                else:
                    pass
            # self.qoe = np.average(self.QoE) # we only need it for each hour.
        elif settings.market_design == "community":
            self.QoE = np.nan * np.ones(settings.nr_of_h)
            # raise Warning("community shadow price and QoE not implemented yet \n")

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
                    aux=[]
                    for agent2 in agent_data.agent_name:
                        aux.append(self.shadow_price[t][agent][agent2]*self.Gn[agent][t] - self.shadow_price[t][agent][agent2]* self.Ln[agent][t])
                    self.settlement[agent][t] = sum(aux)
                    
        elif settings.market_design == "pool":
            if settings.network_type is None:
                for t in range(0, settings.nr_of_h):
                    for agent in agent_data.agent_name:
                        aux = []
                        for agent2 in agent_data.agent_name: # TODO check this!
                            aux.append(self.shadow_price['uniform price'][t]*self.Gn[agent][t] - self.shadow_price['uniform price'][t]* self.Ln[agent][t])
                        self.settlement[agent][t] = sum(aux)
            elif settings.network_type == "direction":
                for t in range(settings.nr_of_h):
                    for agent in agent_data.agent_name:
                        self.settlement[agent][t] = self.shadow_price[agent][t] * self.Pn[agent][t]
            elif settings.network_type == "size":
                raise NotImplementedError("network-aware size is not implemented yet")

    # a function working on the result object, to create plots
    def plot_market_clearing(self, period: int, settings: MarketSettings, agent_data: AgentData, outfile: str):
        """
        makes a plot in the file named at path outfile.
        """
        if not period <= settings.nr_of_h:
            raise ValueError("period should be in range(settings.nr_of_h)")

        if self.market == "pool" and settings.network_type is None:
            data1, data2, yy_gen, yy_load = prep_plot_market_clearing_pool(period, agent_data)
            # Plotting
            plt.step(data1, yy_gen)  # Generation curve
            plt.step(data2, yy_load)  # Load curve
            plt.plot(np.sum(self.Ln.T[period]), abs(self.shadow_price.T[period]), 'ro')  # (heat negotiated,shadow price)
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
            filename = self.market + today.strftime("%d%m%Y_%H:%M:%S") + ".pickle"
            path_to_file = "./pickled_data/" + filename

        # open a file, where you want to store the data
        file = open(path_to_file, 'wb')
        # dump information to that file
        pickle.dump(self, file)
        # close the file
        file.close()

    def convert_to_dicts(self):
        print("todo")
