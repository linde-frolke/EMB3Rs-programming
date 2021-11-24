import cvxpy as cp
import numpy as np

from ...short_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder


def add_energy_budget(constraint_builder, load_var, agent_data):
    """
    :param constraint_builder A constraintBuilder object.
    :param load_var     A cp variable of dimension (hrs, agents)
    :param total_budget     A cp.parameter of size (1, agents) or (, agents) with the load budget for each agent.
                            This would have to be set to zero for producers.
    :return: the constraintBuilder with
    """
    # decide on total budget
    total_budget = cp.Parameter(agent_data.nr_of_agents,
                                value=np.sum(0.5 * (agent_data.lmin + agent_data.lmax).to_numpy(), axis=0))

    # adapts constraintBuilder by adding the energy budget constraint for each load
    constraint_builder.add_constraint(
        cp.sum(load_var, axis=0) == total_budget, str_="energyBudget")

    return constraint_builder
