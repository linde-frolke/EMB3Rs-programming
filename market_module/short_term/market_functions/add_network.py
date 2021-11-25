import cvxpy as cp
import numpy as np

from ...short_term.constraintbuilder.ConstraintBuilder import ConstraintBuilder


def add_network_directions(constraint_builder, settings, network_data, Pn_var):
    """
    :param constraint_builder A constraintBuilder object.
    :param settings     Object of type MarketSettings.
    :param network_data    An object of type Network, containing set of nodes and edges, and the node for each agent.
    :param Pn_var    A cp.variable of size (hours, agents) containing the power injection variable for each agent.

    :return: the constraintBuilder with added power flow direction constraints.
    """
    if settings.network_type == "direction":
        # define network flows
        Ppipe = cp.Variable((settings.nr_of_h, network_data.nr_of_p))

        # define total nodal power injections
        Pnode = cp.Variable((settings.nr_of_h, network_data.nr_of_n))
        for t in settings.timestamps:
            for n in range(network_data.nr_of_n):
                select_agents = np.where(
                    network_data.loc_a == network_data.N[n])
                constraint_builder.add_constraint(Pnode[t, n] == cp.sum(Pn_var[t, select_agents]),
                                                  str_="def_nodal_P" + str(t) + "_" + str(network_data.N[n]))

        # add flow continuity constraint relating nodal and pipeline power flows
        for t in settings.timestamps:
            constraint_builder.add_constraint(
                network_data.A @ Ppipe[t, :] == Pnode[t, :], str_="flow_continuity")

        # adapt constraintBuilder by adding pipeline flow restrictions
        constraint_builder.add_constraint(
            Ppipe >= 0, str_="unidirectional_pipeline_flow")
    elif settings.network_type == "size":
        raise NotImplemented("need to implement this")

    return constraint_builder
