import cvxpy as cp
import numpy as np
from math import ceil
import pandas as pd
try:
    from copt_cvxpy import *
    COPT_INSTALLED = True
except ModuleNotFoundError:
    COPT_INSTALLED = False
    print("COPT not installed in this environment")


from ...short_term.datastructures.resultobject import ResultData
from ...short_term.datastructures.inputstructs import AgentData, MarketSettings
from ...short_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder
from ...short_term.market_functions.add_energy_budget import add_energy_budget
from ...short_term.market_functions.add_network import add_network_directions


def make_pool_market(agent_data: AgentData, settings: MarketSettings, network=None):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: str
    :param agent_data:
    :param settings:
    :param network: an object of class Network, or None if the network data is not needed.
    :return: ResultData object.
    """
    prob_stat = True
    day_nr = []

    ## ITERATE PER DAY 
    h_per_iter = 24 
    nr_of_iter = ceil(settings.nr_of_h / h_per_iter)
    iter_days = range(nr_of_iter)
    h_on_last = settings.nr_of_h - (nr_of_iter - 1)*h_per_iter 

    ## store variables here
    Pn_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), columns=agent_data.agent_name)
    Ln_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), columns=agent_data.agent_name)
    Gn_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), columns=agent_data.agent_name)
    if settings.network_type == "direction":
        shadow_price_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), columns=agent_data.agent_name)
    else:
        shadow_price_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), columns=['uniform price'])

    # convert gmin gmax etc
    gmin = agent_data.gmin.to_numpy()
    gmax = agent_data.gmax.to_numpy()
    lmin = agent_data.lmin.to_numpy()
    lmax = agent_data.lmax.to_numpy()
    cost_new = agent_data.cost.to_numpy()
    util_new = agent_data.util.to_numpy()

    for iter_ in iter_days:
        print("running market for day " + str(iter_ + 1) + " of " + str(nr_of_iter))
        # set the number of timesteps in this iteration
        if iter_ == (nr_of_iter - 1):
            nr_of_timesteps = h_on_last
        else:
            nr_of_timesteps = h_per_iter

        selected_timesteps = range(iter_*h_per_iter, iter_*h_per_iter + nr_of_timesteps)


        # collect named constraints in cb
        cb = ConstraintBuilder()

        # prepare parameters
        Gmin = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=gmin[selected_timesteps, :])
        Gmax = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=gmax[selected_timesteps, :])
        Lmin = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=lmin[selected_timesteps, :])
        Lmax = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=lmax[selected_timesteps, :])
        cost = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=cost_new[selected_timesteps, :])
        util = cp.Parameter(
            (nr_of_timesteps, agent_data.nr_of_agents), value=util_new[selected_timesteps, :])

        # variables
        Pn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Pn")
        Gn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Gn")
        Ln = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Ln")

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
        if settings.network_type is None:
            cb.add_constraint(-cp.sum(Pn, axis=1) == 0, str_="powerbalance")

        # objective function
        # cp.multiply is element-wise multiplication
        total_cost = cp.sum(cp.multiply(cost, Gn))
        total_util = cp.sum(cp.multiply(util, Ln))
        objective = cp.Minimize(total_cost - total_util)

        # add block offers
        if settings.offer_type == "block":
            # Binary variable
            b = cp.Variable(
                (nr_of_timesteps, agent_data.nr_of_agents), boolean=True, name="b")

            for agent in agent_data.block:
                for j in agent_data.block[agent]:
                    for hour in j:
                        # agent_ids.index(agent)->getting the agent's index
                        cb.add_constraint(Gn[hour, agent_data.agent_name.index(agent)] ==
                                        Gmax[hour][agent_data.agent_name.index(agent)] *
                                        b[hour, agent_data.agent_name.index(agent)], str_='block_constraint1')
                        cb.add_constraint(cp.sum(b[j, agent_data.agent_name.index(agent)]) ==
                                        len(j)*b[j[0], agent_data.agent_name.index(agent)], str_='block_constraint2')
    
        # add extra constraint if offer type is energy Budget.
        if settings.offer_type == "energyBudget":
            # add energy budget.
            tot_budget = np.sum(0.5 * (lmin[selected_timesteps, :] + lmax[selected_timesteps, :]), axis=0)
            cb = add_energy_budget(cb, load_var=Ln, total_budget=tot_budget, agent_data=agent_data)

        # add network constraints if this is in the settings
        if settings.network_type is not None:
            if network is None:
                raise ValueError(
                    "You need to give a Network object as input, if you want to include network constraints")
            else:
                cb = add_network_directions(cb, settings, network, Pn, nr_of_timesteps)

        # common for all offer types ------------------------------------------------
        # define the problem and solve it.
        prob = cp.Problem(objective, constraints=cb.get_constraint_list())
        if settings.offer_type == "block":
            result_ = prob.solve(solver=cp.SCIP)
        else:
            if settings.solver == 'GUROBI':
                result_ = prob.solve(solver=cp.GUROBI)
            elif settings.solver == 'SCIP':
                result_ = prob.solve(solver=cp.SCIP)
            elif settings.solver == 'HIGHS':
                result_ = prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
            elif settings.solver == 'COPT':
                if not COPT_INSTALLED:
                    raise Exception("Solver COPT is not installed for usage")
                result_ = prob.solve(solver=COPT())
        print("problem status: %s" % prob.status)

        # throw an error if the problem is not solved.
        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Optimal value: %s" % prob.value)
        else:
            print("Problem is %s" % prob.status) 
            prob_stat = False
            day_nr += int(iter_) 
            # raise RuntimeError("the problem on day " + str(iter_) + " is " + prob.status)
            # raise ValueError("Given your inputs, the problem is %s" % prob.status)


        # COMPUTE SHADOW PRICE -------------------------------------------------------------------
        if settings.offer_type == 'block':
            shadow_price = []
            for t in settings.timestamps:
                max_cost_disp = []
                for agent in agent_data.agent_name:
                    if Gn[agent][t] > 0:
                        max_cost_disp.append(agent_data.cost[agent][t])
                if len(max_cost_disp) > 0:
                    shadow_price.append(max(max_cost_disp))
                else:  # if there is no generation
                    shadow_price.append(min(agent_data.cost.T[t]))
            shadow_price = pd.DataFrame(
                shadow_price, columns=["uniform price"])
        elif settings.network_type is not None:
            if settings.network_type == "direction":  # in this case we have nodal prices
                nodal_prices = np.zeros(
                    (nr_of_timesteps, network.nr_of_n))
                for t in range(nr_of_timesteps):
                    for n in range(network.nr_of_n):
                        nodal_prices[t, n] = cb.get_constraint(str_="def_nodal_P" + str(t) + "_" +
                                                                    str(network.N[n])).dual_value
                # shadow_price = np.array((nr_of_timesteps, agent_data.nr_of_agents))
                mapping = [network.N.index(x) for x in network.loc_a]
                shadow_price = nodal_prices[:, mapping]
                shadow_price = pd.DataFrame(shadow_price, columns=agent_data.agent_name)
            if settings.network_type == "size":  # in this case we have nodal prices
                raise NotImplementedError("this is to be implemented")
        # if no special case of pool market, the output is simple:
        else:
            shadow_price = cb.get_constraint(str_="powerbalance").dual_value
            shadow_price = pd.DataFrame(shadow_price, columns=["uniform price"])


        # store result in result object ---------------------------------------------------------
        Pn_t.iloc[selected_timesteps] = list(Pn.value)
        Ln_t.iloc[selected_timesteps] = list(Ln.value)
        Gn_t.iloc[selected_timesteps] = list(Gn.value)
        shadow_price_t.iloc[selected_timesteps] = list(shadow_price.values)
    
    
    result = ResultData(prob_status=prob_stat, day_nrs=day_nr, 
                        Pn_t=Pn_t, Ln_t=Ln_t, Gn_t=Gn_t, shadow_price_t=shadow_price_t, 
                        agent_data=agent_data, settings=settings)

    return result
