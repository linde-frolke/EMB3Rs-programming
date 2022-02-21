import cvxpy as cp
import numpy as np

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
    # collect named constraints in cb
    cb = ConstraintBuilder()

    # prepare parameters
    Gmin = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.gmin.to_numpy())
    Gmax = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.gmax.to_numpy())
    Lmin = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.lmin.to_numpy())
    Lmax = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.lmax.to_numpy())

    cost = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.cost.to_numpy())
    util = cp.Parameter(
        (settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.util.to_numpy())

    # variables
    Pn = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Pn")
    Gn = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Gn")
    Ln = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Ln")

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
            (settings.nr_of_h, agent_data.nr_of_agents), boolean=True, name="b")

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
        cb = add_energy_budget(cb, load_var=Ln, agent_data=agent_data)

    # add network constraints if this is in the settings
    if settings.network_type is not None:
        if network is None:
            raise ValueError(
                "You need to give a Network object as input, if you want to include network constraints")
        else:
            cb = add_network_directions(cb, settings, network, Pn)

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

    # store result in result object
    result = ResultData(prob, cb, agent_data,
                        settings, network_data=network)

    return result
