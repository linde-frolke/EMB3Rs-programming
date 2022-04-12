import cvxpy as cp
import numpy as np
import pandas as pd
from ...long_term.datastructures.resultobject import ResultData
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings
from ...long_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder


def make_centralized_market(agent_data: AgentData, settings: MarketSettings):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: str
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """



    Pn_t = pd.DataFrame(0.0, index=np.arange(agent_data.day_range*settings.recurrence*agent_data.data_size), columns=agent_data.agent_name)
    Ln_t = pd.DataFrame(0.0, index=np.arange(agent_data.day_range*settings.recurrence*agent_data.data_size), columns=agent_data.agent_name)
    Gn_t = pd.DataFrame(0.0, index=np.arange(agent_data.day_range*settings.recurrence*agent_data.data_size), columns=agent_data.agent_name)
    shadow_price_t = pd.DataFrame(0.0, index=np.arange(agent_data.day_range*settings.recurrence*agent_data.data_size), columns=['uniform price'])

    #simplifying simulation time
    t = agent_data.day_range * settings.recurrence * agent_data.data_size

    for n_iter in range(0, t):
        # collect named constraints in cb
        cb = ConstraintBuilder()
        # prepare parameters
        Gmin = cp.Parameter((agent_data.nr_of_agents), value=agent_data.gmin.T[n_iter].to_numpy())
        Gmax = cp.Parameter((agent_data.nr_of_agents), value=agent_data.gmax.T[n_iter].to_numpy())
        Lmin = cp.Parameter((agent_data.nr_of_agents), value=agent_data.lmin.T[n_iter].to_numpy())
        Lmax = cp.Parameter((agent_data.nr_of_agents), value=agent_data.lmax.T[n_iter].to_numpy())

        cost = cp.Parameter((agent_data.nr_of_agents), value=agent_data.cost.T[n_iter].to_numpy())
        util = cp.Parameter((agent_data.nr_of_agents), value=agent_data.util.T[n_iter].to_numpy())

        # variables
        Pn = cp.Variable((agent_data.nr_of_agents), name="Pn")
        Gn = cp.Variable((agent_data.nr_of_agents), name="Gn")
        Ln = cp.Variable((agent_data.nr_of_agents), name="Ln")

        # variable limits ----------------------------------
        #  Equality and inequality constraints are elementwise, whether they involve scalars, vectors, or matrices.
        cb.add_constraint(Gmin <= Gn, str_="G_lb")
        cb.add_constraint(Gn <= Gmax, str_="G_ub")
        cb.add_constraint(Lmin <= Ln, str_="L_lb")
        cb.add_constraint(Ln <= Lmax, str_="L_ub")

        # constraints --------------------------------------
        # define power injection as net generation
        cb.add_constraint(Pn == Gn - Ln, str_="def_P")

        # power balance at each time - a list of n_t constraints
        cb.add_constraint(cp.sum(Pn) == 0, str_="powerbalance")

        # objective function
        # cp.multiply is element-wise multiplication
        total_cost = cp.sum(cp.multiply(cost, Gn))
        total_util = cp.sum(cp.multiply(util, Ln))
        objective = cp.Minimize(total_cost - total_util)

        # common for all offer types ------------------------------------------------
        # define the problem and solve it.
        prob = cp.Problem(objective, constraints=cb.get_constraint_list())
        result_ = prob.solve(solver=cp.SCIP)
        print("problem status: %s" % prob.status)

        # throw an error if the problem is not solved.
        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Optimal value: %s" % prob.value)
        else:
            # print("Problem is %s" % prob.status)
            raise ValueError("Given your inputs, the problem is %s" % prob.status)

        variables = prob.variables()
        varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]
        Pn_t.T[n_iter] = list(variables[varnames.index("Pn")].value)
        Ln_t.T[n_iter] = list(variables[varnames.index("Ln")].value)
        Gn_t.T[n_iter] = list(variables[varnames.index("Gn")].value)
        shadow_price_t.T[n_iter] = cb.get_constraint(str_="powerbalance").dual_value

    # store result in result object
    result = ResultData(prob, cb, agent_data, settings, Pn_t, Ln_t, Gn_t, shadow_price_t)

    return result
