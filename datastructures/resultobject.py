import cvxpy as cp
import numpy as np
import pandas as pd
import statistics as st
import matplotlib.pyplot as plt
from datastructures.inputstructs import AgentData, MarketSettings
from constraintbuilder.ConstraintBuilder import ConstraintBuilder



class ResultData:
    def __init__(self, name, prob: cp.problems.problem.Problem,
                 cb: ConstraintBuilder,
                 agent_data: AgentData, settings: MarketSettings):
        """
        Object to store relevant outputs from a solved market problem.
        Computes them if needed.
        :param name: str
        :param prob: cvxpy problem object
        :param cb: ConstraintBuilder used for cvxpy problem
        :param agent_data: AgentData object
        :param settings: MarketSettings object
        """
        #
        self.name = name
        self.Market = settings.market_design

        if prob.status in ["infeasible", "unbounded"]:
            self.optimal = False
            raise Warning("problem is not solved to an optimal solution. result object will not contain any info")

        else:
            self.optimal = True
            # store values of the optimized variables
            variables = prob.variables()
            varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]
            self.Pn = pd.DataFrame(variables[varnames == "Pn"].value, columns=agent_data.agent_name)
            self.Ln = pd.DataFrame(variables[varnames == "Ln"].value, columns=agent_data.agent_name)
            self.Gn = pd.DataFrame(variables[varnames == "Gn"].value, columns=agent_data.agent_name)
            if settings.market_design == "p2p":
                # extract trade variable - a square dataframe for each time index
                self.Tnm = [pd.DataFrame(variables[varnames.index("Tnm_" + str(t))].value,
                                         columns=agent_data.agent_name, index=agent_data.agent_name)
                            for t in range(settings.nr_of_h)]
            
                
            
            # get dual of powerbalance for each time
            if settings.market_design == "pool":
                self.shadow_price = cb.get_constraint(str_="powerbalance").dual_value
                self.shadow_price = pd.DataFrame(self.shadow_price, columns=["uniform price"])
                #QOE
                self.qoe= '-'
                #Plotting
                hour=0 #hour to plot
                #Consumers
                data2=[] #Sorted and cumulative producers heat offer
                yy_load=[] #Sorted producers offer
                #sorting according to the index (reverse)
                for i in np.argsort(agent_data.util.T[hour])[::-1]: 
                    data2.append(agent_data.lmax.T[hour][i])
                    yy_load.append(agent_data.util.T[hour][i])
                data2=np.cumsum(data2)
                data2=np.insert(data2,0,0)
                data2=np.insert(data2,-1,data2[-1])
                
                yy_load=np.insert(yy_load,0,yy_load[0])
                yy_load=np.insert(yy_load,-1,yy_load[-1])
                
                # Producers
                data1=[] #Sorted and cumulative producers heat offer
                yy_gen=[] #Sorted producers offer
                #sorting according to the index
                for i in np.argsort(agent_data.cost.T[hour]): 
                    data1.append(agent_data.gmax.T[hour][i])
                    yy_gen.append(agent_data.cost.T[hour][i])
                data1=np.cumsum(data1)
                data1=np.insert(data1,0,0)
                data1=np.insert(data1,-1,data1[-1])
                yy_gen=np.insert(yy_gen,0,0)
                yy_gen=np.insert(yy_gen,-1,yy_gen[-1])
                
                #Plotting
                plt.step(data1,yy_gen) #Generation curve
                plt.step(data2,yy_load) #Load curve
                plt.plot(sum(self.Pn.T[hour]),abs(self.shadow_price.T[hour]),'ro') #(heat negotiated,shadow price)
                plt.ylabel('Price (€/kWh)')
                plt.xlabel('Heat (kWh)')
                plt.xlim([0, max(data2)*1.1])
                plt.ylim([0, max(yy_load)*1.1])
                
            elif settings.market_design == "p2p":
                self.shadow_price = pd.DataFrame(index=settings.timestamps, columns=agent_data.agent_name)
                for t in settings.timestamps:
                    self.shadow_price.iloc[t, :] = cb.get_constraint(str_="p2p_balance_t" + str(t)).dual_value
                    
                #QoE
                self.QoE=[]
                for t in range(0,settings.nr_of_h):
                    lambda_j=[]
                    for a1 in agent_data.agent_name:
                        for a2 in agent_data.agent_name:
                            if self.Pn[a1][t]!=0: #avoid #DIV/0! error 
                                lambda_j.append(agent_data.cost[a1][t]*self.Tnm[t][a1][a2]/self.Pn[a1][t])
                            if self.Ln[a1][t]!=0: #avoid #DIV/0! error 
                                lambda_j.append(agent_data.util[a1][t]*self.Tnm[t][a1][a2]/self.Ln[a1][t])
                    if ((max(lambda_j)-min(lambda_j)))!=0:  #avoid #DIV/0! error        
                        self.QoE.append(1-(st.pstdev(lambda_j)/((max(lambda_j)-min(lambda_j)))))  
                    else:
                        pass
                self.qoe = np.average(self.QoE)
                
                
            elif settings.market_design == "community":
                raise ValueError("not implemented yet \n")

            # hourly social welfare an array of length settings.nr_of_h
            # TODO compute social welfare for each hour
            social_welfare = np.zeros(settings.nr_of_h)
            self.social_welfare_h = social_welfare
            # total social welfare
            self.social_welfare_tot = prob.solution.opt_val
            # scheduled power injection for each agent, for each time
            # agree on sign when load / supply

            # joint market properties

            #OF not considering penalties; Must be equal to social_welfare except when considering preferences
            settlement = np.sum(np.sum(np.multiply(agent_data.cost, self.Gn))) - np.sum(np.sum(np.multiply(agent_data.util, self.Ln)))
            
            self.joint = pd.DataFrame([self.qoe, settlement, self.social_welfare_tot],
                                      columns=[settings.market_design],
                                      index=["QoE", "settlement", "Social Welfare"])

            
            
                        
            