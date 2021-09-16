import cvxpy as cp
import numpy as np
from datastructures.resultobject import ResultData
from datastructures.inputstructs import AgentData, MarketSettings, Network
from constraintbuilder.ConstraintBuilder import ConstraintBuilder
from market_functions.add_energy_budget import add_energy_budget
import itertools
from pyscipopt.scip import Model


def make_p2p_market(name: str, agent_data: AgentData, settings: MarketSettings, network: Network):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: string, can give the resulting ResultData object a name.
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """
    # collect named constraints in cb
    cb = ConstraintBuilder()


# prepare parameters
    Gmin = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.gmin.to_numpy())
    Gmax = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.gmax.to_numpy())
    Lmin = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.lmin.to_numpy())
    Lmax = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.lmax.to_numpy())

    cost = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.cost.to_numpy())
    util = cp.Parameter((settings.nr_of_h, agent_data.nr_of_agents), value=agent_data.util.to_numpy())

    # variables
    Pn = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Pn")
    Gn = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Gn")
    Ln = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents), name="Ln")
    # trades. list of matrix variables, one for each time step.
    Tnm = [cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents),
                       name="Tnm_" + str(t)) for t in settings.timestamps]
    Snm = [cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents),
                       name="Snm_" + str(t)) for t in settings.timestamps]
    Bnm = [cp.Variable((agent_data.nr_of_agents, agent_data.nr_of_agents),
                       name="Bnm_" + str(t)) for t in settings.timestamps]

    # variable limits -----------------------------
    #  Equality and inequality constraints are element-wise, whether they involve scalars, vectors, or matrices.
    cb.add_constraint(Gmin <= Gn, str_="G_lb")
    cb.add_constraint(Gn <= Gmax, str_="G_ub")
    cb.add_constraint(Lmin <= Ln, str_="L_lb")
    cb.add_constraint(Ln <= Lmax, str_="L_ub")
    # limits on trades
    for t in settings.timestamps:
        cb.add_constraint(0 <= Bnm[t], str_="B_lb_t" + str(t))
        cb.add_constraint(0 <= Snm[t], str_="S_lb_t" + str(t))
        cb.add_constraint(Tnm[t] == Snm[t] - Bnm[t], str_="def_S_B_t" + str(t))
        # cannot sell more than I generate
        # cb.add_constraint(cp.sum(Snm[t], axis=1) <= Gn[t, :], str_="S_ub_t" + str(t))
        # cannot buy more than my load
        # cb.add_constraint(cp.sum(Bnm[t], axis=1) <= Ln[t, :], str_="S_ub_t" + str(t))

    # constraints ----------------------------------
    # define relation between generation, load, and power injection
    cb.add_constraint(Pn == Gn - Ln, str_="def_P")
    for t in settings.timestamps:
        # trade reciprocity
        for i, j in itertools.product(range(agent_data.nr_of_agents), range(agent_data.nr_of_agents)):
            # if not i == j:
            if j >= i:
                cb.add_constraint(Tnm[t][i, j] + Tnm[t][j, i] == 0, str_="reciprocity_t" + str(t) + str(i) + str(j))
        # total trades have to match power injection
        cb.add_constraint(Pn[t, :] == cp.sum(Tnm[t], axis=1), str_="p2p_balance_t" + str(t))

    # add extra constraint if offer type is energy Budget.
    if settings.offer_type == "energyBudget":
        # add energy budget.
        cb = add_energy_budget(cb, load_var=Ln, agent_data=agent_data)
        
    if settings.offer_type == "block":
        #Binary variable
        b = cp.Variable((settings.nr_of_h, agent_data.nr_of_agents),boolean=True ,name="b")
        
        for agent in agent_data.block:
            for j in agent_data.block[agent]:
                for hour in j:
                    #agent_ids.index(agent)->getting the agent's index
                    cb.add_constraint(Gn[hour,agent_data.agent_name.index(agent)] == Gmax[hour][agent_data.agent_name.index(agent)]*b[hour,agent_data.agent_name.index(agent)], str_='block_constraint1') 
                    cb.add_constraint(cp.sum(b[j,agent_data.agent_name.index(agent)]) == len(j)*b[j[0],agent_data.agent_name.index(agent)], str_='block_constraint2')
#
    # objective function
    total_cost = cp.sum(cp.multiply(cost, Gn))  # cp.multiply is element-wise multiplication
    total_util = cp.sum(cp.multiply(util, Ln))
    # make different objfun depending on preference settings
    if settings.product_diff == "noPref":
        objective = cp.Minimize(total_cost - total_util)
    else:
        # construct preference matrix
        # TODO could move this to AgentData structure
        if settings.product_diff == "co2Emissions": 
            emissions_p = agent_data.co2_emission/sum(agent_data.co2_emission.T[0]) #percentage
            emissions_p = np.tile(emissions_p,(len(agent_data.agent_name),1))
            for t in settings.timestamps:
                co2_penalty = cp.sum(cp.multiply(np.array(emissions_p), Snm[t]))
            objective = cp.Minimize(total_cost - total_util + co2_penalty )
            
        if settings.product_diff == "networkDistance":
            for t in settings.timestamps:    
                distance_penalty = cp.sum(cp.multiply(network.all_distance_percentage, Snm[t]))
            objective = cp.Minimize(total_cost - total_util + distance_penalty)

        if settings.product_diff == "losses":
            for t in settings.timestamps:    
                losses_penalty = cp.sum(cp.multiply(network.all_losses_percentage, Snm[t]))

            objective = cp.Minimize(total_cost - total_util + losses_penalty)

    # define the problem and solve it.
    prob = cp.Problem(objective, constraints=cb.get_constraint_list())
    result_ = prob.solve(solver=cp.SCIP)
    
 

    print("problem status: %s" % prob.status)
 
    if prob.status not in ["infeasible", "unbounded"]:
        # Otherwise, problem.value is inf or -inf, respectively.
        print("Optimal value: %s" % prob.value)
    else:
        print("Problem status is %s" % prob.status)
 
    # store result in result object
    result = ResultData(name, prob, cb, agent_data, settings)
 
    return result
