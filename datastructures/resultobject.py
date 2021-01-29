import cvxpy as cp
import numpy as np
import pandas as pd
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

            # get dual of powerbalance for each time
            # TODO perhaps this is different for different market types
            # TODO make data frame?
            self.shadow_price = cb.get_constraint(str_="powerbalance").dual_value

            # hourly social welfare an array of length settings.nr_of_h
            # TODO compute social welfare for each hour
            social_welfare = np.zeros(settings.nr_of_h)
            self.social_welfare_h = social_welfare
            # total social welfare
            self.social_welfare_tot = sum(social_welfare)
            # scheduled power injection for each agent, for each time
            # agree on sign when load / supply

            # joint market properties
            qoe = 100.0  # TODO compute it
            settlement = 5.0  # TODO compute it
            self.joint = pd.DataFrame([qoe, settlement, self.social_welfare_tot],
                                      columns=[settings.market_design],
                                      index = ["QoE", "settlement", "Social Welfare"])


