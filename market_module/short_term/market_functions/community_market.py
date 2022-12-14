import cvxpy as cp
import numpy as np
from math import ceil
import pandas as pd

from ...short_term.datastructures.resultobject import ResultData
from ...short_term.datastructures.inputstructs import AgentData, MarketSettings
from ...short_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder
from ...short_term.market_functions.add_energy_budget import add_energy_budget


def make_community_market(agent_data: AgentData, settings: MarketSettings):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: str
    :param agent_data:
    :param settings:
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
    shadow_price_t = pd.DataFrame(0.0, index=np.arange(settings.nr_of_h), 
                                    columns=["community", "export", "import", "non-community"])
    
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

        # params particular to community
        gamma_exp = cp.Parameter(value=settings.gamma_exp, nonpos=True)
        gamma_imp = cp.Parameter(value=settings.gamma_imp, nonneg=True)

        # variables
        Pn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Pn")
        Gn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Gn")
        Ln = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Ln")

        # variables particular to community
        q_imp = cp.Variable(nr_of_timesteps, name="q_imp")
        q_exp = cp.Variable(nr_of_timesteps, name="q_exp")
        if settings.community_objective == "peakShaving":
            q_peak = cp.Variable(name="q_peak")

        # Community import variable
        alpha = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents),
                            name="alpha", nonneg=True)  
        # Agents outside the community cannot import
        cb.add_constraint(alpha[:, agent_data.notC] == 0, str_="set_alpha_zero")  
        # Export variable
        beta = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents),
                        name="beta", nonneg=True)  
        # Agents outside the community cannot export
        cb.add_constraint(beta[:, agent_data.notC] == 0, str_="set_beta_zero") 

        # sale to / buy from community (negative if bought)
        s_comm = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="s_comm")
        # set s_comm to zero for agents not in the community
        cb.add_constraint(s_comm[:, agent_data.notC] == 0, str_="set_beta_zero")  

        # variable limits ----------------------------------
        #  Equality and inequality constraints are elementwise, whether they involve scalars, vectors, or matrices.
        cb.add_constraint(Gmin <= Gn, str_="G_lb")
        cb.add_constraint(Gn <= Gmax, str_="G_ub")
        cb.add_constraint(Lmin <= Ln, str_="L_lb")
        cb.add_constraint(Ln <= Lmax, str_="L_ub")

        # limits particular to community market
        cb.add_constraint(q_imp >= 0, str_="q_imp_lb")
        cb.add_constraint(q_exp >= 0, str_="q_exp_lb")

        # constraints --------------------------------------
        # define power injection as net generation
        cb.add_constraint(Pn == Gn - Ln, str_="def_P")

        # constraints particular to community market
        # agents outside the community power balance
        cb.add_constraint(q_imp - q_exp == sum(Pn[:, i] for i in agent_data.notC), str_="noncom_powerbalance")

        # net injection is sold to community, bought from community, or sold/bought from outside
        cb.add_constraint(Pn[:, agent_data.C] == s_comm[:, agent_data.C] - alpha[:, agent_data.C] +
                          beta[:, agent_data.C], str_="comm_agent_powerbalance")
        # internal trades add up to zero
        cb.add_constraint(-cp.sum(s_comm, axis=1) == 0, str_="internal_trades")
        # define total import and export
        cb.add_constraint(cp.sum(alpha, axis=1) == q_imp, str_="total_imp")
        cb.add_constraint(q_exp == cp.sum(beta, axis=1), str_="total_exp")

        if settings.community_objective == "peakShaving":
            cb.add_constraint(q_imp <= q_peak, str_="def_peak")

        # objective function
        # cp.multiply is element-wise multiplication
        total_cost = cp.sum(cp.multiply(cost, Gn))
        total_util = cp.sum(cp.multiply(util, Ln))
        if settings.community_objective == "peakShaving":
            punished_peak = settings.gamma_peak * q_peak
            objective = cp.Minimize(total_cost - total_util + punished_peak)
        elif settings.community_objective == "autonomy":
            punished_imp_exp = gamma_imp * \
                cp.sum(q_imp) + gamma_exp * cp.sum(q_exp)
            objective = cp.Minimize(total_cost - total_util + punished_imp_exp)

        # add extra constraint if offer type is energy Budget.
        if settings.offer_type == "energyBudget":
            # add energy budget.
            tot_budget = np.sum(0.5 * (lmin[selected_timesteps, :] + lmax[selected_timesteps, :]), axis=0)
            cb = add_energy_budget(cb, load_var=Ln, total_budget=tot_budget, agent_data=agent_data)

        # common for all offer types ------------------------------------------------
        # define the problem and solve it.
        prob = cp.Problem(objective, constraints=cb.get_constraint_list())
        result_ = prob.solve(solver=cp.GUROBI)
        print("problem status: %s" % prob.status)

        # throw an error if the problem is not solved.
        if prob.status not in ["infeasible", "unbounded"]:
            # Otherwise, problem.value is inf or -inf, respectively.
            print("Optimal value: %s" % prob.value)
        else:
            prob_stat = False
            day_nr += int(iter_) 
            print("Given your inputs, the problem is %s" % prob.status)
            

        # compute shadow price
        price_array = np.column_stack([cb.get_constraint(str_="internal_trades").dual_value,
                                                cb.get_constraint(
                                                    str_="total_exp").dual_value,
                                                cb.get_constraint(
                                                    str_="total_imp").dual_value,
                                                cb.get_constraint(str_="noncom_powerbalance").dual_value])
        shadow_price = pd.DataFrame(price_array, columns=["community", "export", "import", "non-community"])

        # store result in result object ---------------------------------------------------------
        Pn_t.iloc[selected_timesteps] = list(Pn.value)
        Ln_t.iloc[selected_timesteps] = list(Ln.value)
        Gn_t.iloc[selected_timesteps] = list(Gn.value)
        shadow_price_t.iloc[selected_timesteps] = list(shadow_price.values)

    # store result in result object
    result = ResultData(prob_status=prob_stat, day_nrs=day_nr, 
                        Pn_t=Pn_t, Ln_t=Ln_t, Gn_t=Gn_t, shadow_price_t=shadow_price_t, 
                        agent_data=agent_data, settings=settings)

    return result
