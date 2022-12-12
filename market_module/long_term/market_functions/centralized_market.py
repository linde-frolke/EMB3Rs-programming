import cvxpy as cp
import numpy as np
import pandas as pd
from ...long_term.datastructures.resultobject import ResultData
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings
from ...long_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder
from math import ceil

def make_centralized_market(agent_data: AgentData, settings: MarketSettings):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: str
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """

    print("running centralized market")
    #simplifying simulation time
    t = settings.diff

    # save variable values in these
    Pn_t = pd.DataFrame(0.0, index=np.arange(t), columns=agent_data.agent_name)
    Ln_t = pd.DataFrame(0.0, index=np.arange(t), columns=agent_data.agent_name)
    Gn_t = pd.DataFrame(0.0, index=np.arange(t), columns=agent_data.agent_name)
    
    shadow_price_t = pd.DataFrame(0.0, index=np.arange(t), columns=['uniform price'])


    # is there any storage present?
    storage_present = len(agent_data.storage_name) > 0
    
    # if no storage, keep as it was
    if not storage_present:
        print("running long term without storage")
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

            if settings.solver == 'GUROBI':
                result_ = prob.solve(solver=cp.GUROBI)
            elif settings.solver == 'SCIP':
                result_ = prob.solve(solver=cp.SCIP)
            elif settings.solver == 'HIGHS':
                result_ = prob.solve(solver=cp.SCIPY, scipy_options={"method": "highs"})
            elif settings.solver == 'COPT':
                result_ = prob.solve(solver=cp.COPT)

            # throw an error if the problem is not solved.
            if prob.status in ["infeasible", "unbounded"]:
                # print("Problem is %s" % prob.status)
                raise ValueError("Given your inputs, the problem is %s" % prob.status)

            variables = prob.variables()
            varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]

            Pn_t.iloc[n_iter] = list(variables[varnames.index("Pn")].value)
            Ln_t.iloc[n_iter] = list(variables[varnames.index("Ln")].value)
            Gn_t.iloc[n_iter] = list(variables[varnames.index("Gn")].value)
            shadow_price_t.iloc[n_iter] = cb.get_constraint(str_="powerbalance").dual_value
           
        result = ResultData(prob, cb, agent_data, settings, Pn_t, Ln_t, Gn_t, shadow_price_t)
        
        # return result
        return result

    # if storage is present, we run the market for each day separately
    else:
        print("running long term with storage")
        En_t = pd.DataFrame(0.0, index=np.arange(t), columns=agent_data.storage_name)
        Bn_t = pd.DataFrame(0.0, index=np.arange(t), columns=agent_data.storage_name)
        # We assume that the storage is empty at the start and end of the day
        E_0 = 0

        ## ITERATE PER DAY 
        h_per_iter = 24 
        nr_of_iter = ceil(t / h_per_iter)
        iter_days = range(nr_of_iter)
        h_on_last = t - (nr_of_iter - 1)*h_per_iter 

        for iter in iter_days:
            print("iteration number " + str(iter+1) + " out of " + str(nr_of_iter) + "\n")
            # set the number of timesteps in this iteration
            if iter == (nr_of_iter - 1):
                nr_of_timesteps = h_on_last
            else:
                nr_of_timesteps = h_per_iter

            selected_timesteps = range(iter*h_per_iter, (iter +1)*h_per_iter)
            # build constraints
            # collect named constraints in cb
            cb = ConstraintBuilder()
            # prepare parameters
            Gmin = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.gmin.to_numpy()[selected_timesteps, :])
            Gmax = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.gmax.to_numpy()[selected_timesteps, :])
            Lmin = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.lmin.to_numpy()[selected_timesteps, :])
            Lmax = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.lmax.to_numpy()[selected_timesteps, :])

            cost = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.cost.to_numpy()[selected_timesteps, :])
            util = cp.Parameter((nr_of_timesteps, agent_data.nr_of_agents), value=agent_data.util.to_numpy()[selected_timesteps, :])

            stor_capacity = cp.Parameter((nr_of_timesteps, agent_data.nr_of_stor), value=agent_data.storage_capacity.to_numpy()[selected_timesteps, :])

            # variables load and generation
            Pn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Pn")
            Gn = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Gn")
            Ln = cp.Variable((nr_of_timesteps, agent_data.nr_of_agents), name="Ln")
            # variables storage
            # the State of Energy of the storage, En
            En = cp.Variable((nr_of_timesteps, agent_data.nr_of_stor), name="En")
            # the amount Charged, Bn
            Bn = cp.Variable((nr_of_timesteps, agent_data.nr_of_stor), name="Bn")

            # variable limits ----------------------------------
            #  Equality and inequality constraints are elementwise, whether they involve scalars, vectors, or matrices.
            cb.add_constraint(Gmin <= Gn, str_="G_lb")
            cb.add_constraint(Gn <= Gmax, str_="G_ub")
            cb.add_constraint(Lmin <= Ln, str_="L_lb")
            cb.add_constraint(Ln <= Lmax, str_="L_ub")
            
            # Storage constraints ---
            # limits 
            cb.add_constraint(En >= 0, str_="En_lb")
            cb.add_constraint(En <= stor_capacity, str_="En_ub")
            # end of day storage level is set to zero
            cb.add_constraint(En[nr_of_timesteps -1 , :] == E_0, str_="set_eod_En")
            # Storage energy balance
            # constraint for first hour of the day. It is assumed that the storage starts and ends at the level E_0
            cb.add_constraint(En[0, :] == Bn[0,:] + E_0, str_="storage_energy_balance_t0")
            for h in range(1, nr_of_timesteps):
                cb.add_constraint(En[h, :] == En[h-1, :] + Bn[h,:], str_="storage_balance_t" + str(h))

            # constraints --------------------------------------
            # define power injection as net generation
            cb.add_constraint(Pn == Gn - Ln, str_="def_P")

            # power balance at each time -- net power injection by generators and loads equals what is charged in the storages.
            cb.add_constraint(cp.sum(Pn, axis=1) == cp.sum(Bn, axis=1), str_="powerbalance")

            # objective function
            # cp.multiply is element-wise multiplication
            total_cost = cp.sum(cp.multiply(cost, Gn))
            total_util = cp.sum(cp.multiply(util, Ln))
            objective = cp.Minimize(total_cost - total_util)

            # common for all offer types ------------------------------------------------
            # define the problem and solve it.
            prob = cp.Problem(objective, constraints=cb.get_constraint_list())
            result_ = prob.solve(solver=cp.GUROBI)

            # throw an error if the problem is not solved.
            if prob.status in ["infeasible", "unbounded"]:
                # print("Problem is %s" % prob.status)
                raise RuntimeError("Given your inputs, the problem is %s" % prob.status)

            variables = prob.variables()
            varnames = [prob.variables()[i].name() for i in range(len(prob.variables()))]

            # save the outputs from this iteration
            Pn_t.iloc[selected_timesteps, :] = variables[varnames.index("Pn")].value
            Ln_t.iloc[selected_timesteps, :] = variables[varnames.index("Ln")].value
            Gn_t.iloc[selected_timesteps, :] = variables[varnames.index("Gn")].value
            En_t.iloc[selected_timesteps, :] = variables[varnames.index("En")].value
            Bn_t.iloc[selected_timesteps, :] = variables[varnames.index("Bn")].value
            
            shadow_price_t.iloc[selected_timesteps] = np.reshape(cb.get_constraint(str_="powerbalance").dual_value, 
                                                                    (h_per_iter, 1))
            # end the iteration    

        # store result in result object
        result = ResultData(prob, cb, agent_data, settings, Pn_t, Ln_t, Gn_t, shadow_price_t, En=En_t, Bn=Bn_t)
        
        # return result
        return result

