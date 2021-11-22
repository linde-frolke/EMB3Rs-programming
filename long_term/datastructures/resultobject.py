import cvxpy as cp
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from datastructures.inputstructs import AgentData, MarketSettings
from constraintbuilder.ConstraintBuilder import ConstraintBuilder
import itertools


class ResultData:
    def __init__(self, name, prob: cp.problems.problem.Problem,
                 cb: ConstraintBuilder,
                 agent_data: AgentData, settings: MarketSettings):
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
            
            if settings.market_design == "decentralized":
                # extract trade variable - a square dataframe for each time index
                self.Tnm = [pd.DataFrame(variables[varnames.index("Tnm_" + str(t))].value,
                                         columns=agent_data.agent_name, index=agent_data.agent_name)
                            for t in range(agent_data.day_range*settings.recurrence*agent_data.data_size)]

            # get values related to duals  ----------------------------------------
            if settings.market_design == "centralized":
                self.shadow_price = cb.get_constraint(str_="powerbalance").dual_value
                self.shadow_price = pd.DataFrame(self.shadow_price, columns=["uniform price"])
            
            
            elif settings.market_design == "decentralized":
           
                self.shadow_price = [pd.DataFrame(index=agent_data.agent_name, columns=agent_data.agent_name)
                                      for t in range(agent_data.day_range*settings.recurrence*agent_data.data_size)]

                for t in range(agent_data.day_range*settings.recurrence*agent_data.data_size):
                    for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
                        # if not i == j:
                        if j >= i:
                            constr_name = "reciprocity_t" + str(t) + str(i) + str(j)
                            self.shadow_price[t].iloc[i, j] = cb.get_constraint(str_=constr_name).dual_value
                            self.shadow_price[t].iloc[j, i] = - self.shadow_price[t].iloc[i, j]

            # initialize empty slots for uncomputed result quantities
            self.QoE = None
            self.social_welfare_h = None
            self.settlement = None
            self.compute_output_quantities(settings, agent_data)
            
            
    # a function to make all relevant output variables
    def compute_output_quantities(self, settings, agent_data):
        # get shadow price, Qoe, for different markets --------------------------------------------------
        if settings.market_design == "centralized":
            self.QoE = np.nan * np.ones(agent_data.day_range*settings.recurrence*agent_data.data_size)
            # raise Warning("QoE not implemented for pool")
        elif settings.market_design == "decentralized":
            # QoE
            self.QoE = pd.DataFrame(index=range(agent_data.day_range*settings.recurrence*agent_data.data_size),columns=["QoE"])
    
            for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
                self.lambda_j = []
                for a1 in agent_data.agent_name:
                    for a2 in agent_data.agent_name:
                        if self.Pn[a1][t] != 0:  # avoid #DIV/0! error
                            self.lambda_j.append(agent_data.cost[a1][t] * self.Tnm[t][a1][a2] / self.Pn[a1][t])
                        if self.Ln[a1][t] != 0:  # avoid #DIV/0! error
                            self.lambda_j.append(agent_data.util[a1][t] * self.Tnm[t][a1][a2] / self.Ln[a1][t])
                
                if len(self.lambda_j)==0: #If no power is traded in t
                    self.QoE["QoE"][t]='Not Defined'
                elif (max(self.lambda_j) - min(self.lambda_j)) != 0:  # avoid #DIV/0! error
                    self.QoE["QoE"][t] = (1 - (st.pstdev(self.lambda_j) / (max(self.lambda_j) - min(self.lambda_j))))
                else:
                    pass
            # self.qoe = np.average(self.QoE) # we only need it for each hour.

        # hourly social welfare an array of length agent_data.day_range*settings.recurrence*agent_data.data_size
        self.social_welfare_h = pd.DataFrame(index=range(agent_data.day_range*settings.recurrence*agent_data.data_size),columns=["Social Welfare"])
        for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
            total_cost = np.sum(np.multiply(agent_data.cost.T[t], self.Gn.T[t]))
            total_util = np.sum(np.multiply(agent_data.util.T[t], self.Ln.T[t]))
            self.social_welfare_h["Social Welfare"][t] = (total_cost - total_util)
        
        #Settlement
        self.settlement = pd.DataFrame(index=range(agent_data.day_range*settings.recurrence*agent_data.data_size),columns=agent_data.agent_name)
        if settings.market_design == "decentralized":
            for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
                for agent in agent_data.agent_name:
                    aux=[]
                    for agent2 in agent_data.agent_name:
                        aux.append(self.shadow_price[t][agent][agent2]*self.Gn[agent][t] - self.shadow_price[t][agent][agent2]* self.Ln[agent][t])
                    self.settlement[agent][t] = sum(aux)
                    
        elif settings.market_design == "centralized":
            for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
                for agent in agent_data.agent_name:
                    aux=[]
                    for agent2 in agent_data.agent_name:
                        aux.append(self.shadow_price['uniform price'][t]*self.Gn[agent][t] - self.shadow_price['uniform price'][t]* self.Ln[agent][t])
                    self.settlement[agent][t] = sum(aux)
                 
        #list with producers+prosumers
        self.prod_pros=[] 
        for i,j in enumerate(agent_data.agent_type.values()):                
            if j == 'prosumer' or j == 'producer':
                self.prod_pros.append(agent_data.agent_name[i])
                
        #AVERAGE DISPATCHED GENERATION (ADG)                  
        self.ADG = pd.DataFrame(index=['ADG'],columns=self.prod_pros)
        for agent in self.prod_pros:
            aux=[]
            for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
                if agent_data.gmax[agent][t] == 0:
                    aux.append(1) #if source production is zero
                else:
                    aux.append(self.Gn[agent][t]/agent_data.gmax[agent][t])
            self.ADG[agent]['ADG'] = np.average(aux)*100
        
        #SUCCESSFUL PARTICIPATION IN THE MARKET (SPM)
        self.SPM = pd.DataFrame(index=['SPM'],columns=self.prod_pros)
        for agent in self.prod_pros:
            aux=[]
            for t in range(0, agent_data.day_range*settings.recurrence*agent_data.data_size):
                if agent_data.gmax[agent][t] == 0:
                    aux.append(0)
                else:
                    if self.Gn[agent][t] != 0: #check small numbers (ex: 1E-15)
                        aux.append(1)
                    else:
                        aux.append(0)
            self.SPM[agent]['SPM'] = np.average(aux)*100
        
        
    

    def find_best_price(self, period: int, agent_name, agent_data, settings):
        if settings.market_design == "decentralized":
            raise ValueError("Find the best price option is only available in centralized market")
        elif period >= agent_data.day_range*settings.recurrence*agent_data.data_size:
            raise ValueError('Please select a period within the simulation time range')
        else:     
            aux=[]
            for name in agent_data.agent_name:
                if self.Gn[name][period] > 0:
                    aux.append([agent_data.cost[name][period], name])
            if len(aux) == 0: 
                print('No production dispatched')
            else:
                self.expensive_prod = max(aux, key=lambda x: x[0])
                return agent_data.cost[self.expensive_prod[1]][period]