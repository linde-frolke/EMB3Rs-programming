import cvxpy as cp
import numpy as np
import pandas as pd
from pyscipopt.scip import Model
import itertools
from ...long_term.datastructures.resultobject import ResultData
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings, Network
from ...long_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder


def make_decentralized_market(agent_data: AgentData, settings: MarketSettings, network: Network):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: string, can give the resulting ResultData object a name.
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """
    Pn_t = pd.DataFrame(0.0, index=np.arange(settings.diff), columns=agent_data.agent_name)
    Ln_t = pd.DataFrame(0.0, index=np.arange(settings.diff), columns=agent_data.agent_name)
    Gn_t = pd.DataFrame(0.0, index=np.arange(settings.diff), columns=agent_data.agent_name)
    shadow_price_t = [pd.DataFrame(0.0, index=agent_data.agent_name, columns=agent_data.agent_name) for t in range(settings.diff)]
    Tnm_t=[pd.DataFrame(0.0, index=agent_data.agent_name, columns=agent_data.agent_name) for t in range(settings.diff)]

    # simplifying simulation time
    t=settings.diff

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

        # trades. list of matrix variables.
        Tnm = cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents), name="Tnm_" + str(n_iter))
        Snm = cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents), name='Snm_' + str(n_iter))
        Bnm = cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents), name='Bnm_' + str(n_iter))

    # variable limits -----------------------------
    #  Equality and inequality constraints are element-wise, whether they involve scalars, vectors, or matrices.
        cb.add_constraint(Gmin <= Gn, str_="G_lb")
        cb.add_constraint(Gn <= Gmax, str_="G_ub")
        cb.add_constraint(Lmin <= Ln, str_="L_lb")
        cb.add_constraint(Ln <= Lmax, str_="L_ub")
    # limits on trades
        cb.add_constraint(0 <= Bnm, str_="B_lb_t" + str(n_iter))
        cb.add_constraint(0 <= Snm, str_="S_lb_t" + str(n_iter))
        cb.add_constraint(Tnm == Snm - Bnm, str_="def_S_B_t" + str(n_iter))
        # cannot sell more than I generate
        cb.add_constraint(cp.sum(Snm, axis=1) <= Gn, str_="S_ub_t" + str(n_iter))
        # cannot buy more than my load
        cb.add_constraint(cp.sum(Bnm, axis=1) <= Ln, str_="S_ub_t" + str(n_iter))
    # constraints ----------------------------------
    # define relation between generation, load, and power injection
        cb.add_constraint(Pn == Gn - Ln, str_="def_P")

        # trade reciprocity
        for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
            # if not i == j:
            if j >= i:
                cb.add_constraint(Tnm[i, j] + Tnm[j, i] == 0,
                                  str_="reciprocity_t" + str(n_iter) + str(i) + str(j))
        # total trades have to match power injection
        cb.add_constraint(Pn == cp.sum(
            Tnm, axis=1), str_="p2p_balance_t" + str(n_iter))

        # objective function
        # cp.multiply is element-wise multiplication
        total_cost = cp.sum(cp.multiply(cost, Gn))
        total_util = cp.sum(cp.multiply(util, Ln))
        # make different objfun depending on preference settings
        if settings.product_diff == "noPref":
            objective = cp.Minimize(total_cost - total_util)
        else:
            # construct preference matrix
            # TODO could move this to AgentData structure
            if settings.product_diff == "co2Emissions":
                emissions_p = agent_data.co2_emission / \
                    sum(agent_data.co2_emission.T[0])  # percentage
                emissions_p = np.tile(emissions_p, (len(agent_data.agent_name), 1))
                for t in range(settings.recurrence):
                    co2_penalty = cp.sum(cp.multiply(
                        np.array(emissions_p), Snm))
                objective = cp.Minimize(total_cost - total_util + co2_penalty)

            if settings.product_diff == "networkDistance":
                for t in range(settings.recurrence):
                    distance_penalty = cp.sum(cp.multiply(
                        network.all_distance_percentage, Snm))
                objective = cp.Minimize(total_cost - total_util + distance_penalty)

        # define the problem and solve it.
        prob = cp.Problem(objective, constraints=cb.get_constraint_list())
        if settings.solver == 'GUROBI':
            result_ = prob.solve(solver=cp.GUROBI)
        elif settings.solver == 'SCIP':
            result_ = prob.solve(solver=cp.SCIP)
        elif settings.solver == 'HIGHS':
            result_ = prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
        elif settings.solver == 'COPT':
            result_ = prob.solve(solver=cp.COPT)

        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Optimal value: %s" % prob.value)
        else:
            print("Problem status is %s" % prob.status)

        variables = prob.variables()
        varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]
        Pn_t.T[n_iter] = list(variables[varnames.index("Pn")].value)
        Ln_t.T[n_iter] = list(variables[varnames.index("Ln")].value)
        Gn_t.T[n_iter] = list(variables[varnames.index("Gn")].value)
        #Getting shadow_price
        for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
            # if not i == j:
            if j >= i:
                constr_name = "reciprocity_t" + \
                              str(n_iter) + str(i) + str(j)
                shadow_price_t[n_iter].iloc[i, j] = cb.get_constraint(str_=constr_name).dual_value
                shadow_price_t[n_iter].iloc[j, i] = - \
                    shadow_price_t[n_iter].iloc[i, j]

        Tnm_t[n_iter] = pd.DataFrame(variables[varnames.index("Tnm_" + str(n_iter))].value,
                                 columns=agent_data.agent_name, index=agent_data.agent_name)


    # store result in result object
    result = ResultData(prob, cb, agent_data, settings, Pn_t, Ln_t, Gn_t, shadow_price_t, Tnm_t)

    return result
