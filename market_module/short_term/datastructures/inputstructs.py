# inputs format for market module

import pandas as pd
import numpy as np
import heapq
from typing import List, Any
from pydantic import BaseModel, validator, Field

# general settings object ---------------------------------------------------------------------------------------------
class MarketSettings(BaseModel):
    """
    Object to store market settings.
    On creation, it checks whether settings are valid
    """
    class Config:
        arbitrary_types_allowed = True
    
    nr_of_h : int  # nr of time steps to run the market for
    market_design : str # market design
    offer_type : str # offer type is either "simple", "block", "energyBudget".
    product_diff : str = "noPref" # product differentiation. Only has an effect if market design = p2p
    el_dependent : bool # should CHP bids be depending on electricity price?
    el_price : np.ndarray = Field(default=None) # a vector with electricity prices. is None by default. 
    network_type : str = None  # whether to include network, and if so, how. Default is None
    # entries to be filled in by other functions
    # filled in by init
    timestamps : Any = Field(default=None)
    elPrice : Any = Field(default=None)
    community_objective : float = None
    gamma_peak : float= None
    gamma_imp : float = None
    gamma_exp : float = None

    def __init__(self, **data) -> None:
        """
        create MarketSettings object if all inputs are correct
        """
        # pydantic __init__ syntax
        super().__init__(**data)
        
        # here we initiate the entries that are computed from inputs
        self.timestamps = np.arange(self.nr_of_h)

        if self. el_dependent:
            self.elPrice = pd.DataFrame(self.el_price, columns=["elprice"])
        else:
            self.elPrice = None

    def add_community_settings(self, objective, g_peak, g_exp, g_imp):
        """ the parameters are optional inputs"""
        # add the options for community to the settings
        options_objective = ["autonomy", "peakShaving"]
        if objective not in options_objective:
            raise ValueError("objective should be one of" + str(options_objective))
        self.community_objective = objective

        # set values of gammas
        if g_peak is None:
            self.gamma_peak = 10.0 ** 2
        else:
            self.gamma_peak = g_peak
        if g_exp is None:
            self.gamma_exp = -4 * 10.0 ** 1
        else:
            if self.gamma_exp >= 0.0:
                raise ValueError("export penalty must be nonpositive")
            self.gamma_exp = g_exp
        if g_imp is None:
            self.gamma_imp = 5 * 10.0 ** 1
        else:
            self.gamma_imp = g_imp

    @validator("nr_of_h")
    def nr_of_hours_check(cls, v):
        max_time_steps = 48  # max 48 hours.
        if not ((type(v) == int) and (1 <= v <= max_time_steps)):
            raise ValueError("nr_of_hours should be an integer between 1 and " + str(max_time_steps))
        return v
    @validator("offer_type")
    def offer_type_valid(cls, v):
        options_offer_type = ["simple", "block", "energyBudget"]
        if v not in options_offer_type:
            raise ValueError("offer_type should be one of " + str(options_offer_type))
        return v
    @validator("product_diff")
    def prof_diff_valid(cls, v):
        options_product_diff = ["noPref", "co2Emissions", "networkDistance", "losses"]
        if v not in options_product_diff:
            raise ValueError('product_diff should be one of ' + str(options_product_diff))
        return v
    @validator("market_design")
    def market_design_valid(cls, v):
        # check if input is correct
        options_market_design = ["pool", "p2p", "community"]
        if v not in options_market_design:
            raise ValueError('market_design should be one of ' + str(options_market_design))
        return v
    @validator("el_dependent")
    def el_price_if_needed(cls, v, values):
        # check inputs for electricity dependence. Can be combined with all 3 market types
        if v:
            if values["el_price"] is None:
                raise ValueError('el_price must be given if el_dependent == True')
            elif not len(values["el_price"]) == values["nr_of_hours"]:
                raise ValueError('el_price must be given for each hour')
        return v
    @validator("network_type")
    def network_type_valid(cls, v, values):
        if v is not None:
            options_network_type = ["direction"]
            if v not in options_network_type:
                raise ValueError("network_type should be None or one of " + str(options_network_type))
            if not values["offer_type"] == "simple":
                raise ValueError("If you want network-awareness, offer_type must be 'simple'")
            if not values["market_design"] == "pool":
                raise ValueError("network-awareness is only implemented for pool, not for p2p and community markets")
    @validator("product_diff")
    def product_diff_only_with_p2p(cls, v, values):
        # exclude bad combination of inputs
        if values["market_design"] != "p2p" and v != "noPref":
            raise ValueError('product_diff can only be something else than "noPref" if market_design == "p2p')
        

# agents information --------------------------------------------------------------------------------------------------
class AgentData:
    """
    Object that stores all agent related data
    Contains an entry for each different input.
    If the input is constant in time, it is a dataframe with agent ID as column name, and the input as row
    If the input is varying in time, it is a dataframe with agent ID as column name, and time along the rows
    """

    def __init__(self, settings, agent_ids, gmax, lmax, cost, util, co2=None,
                 is_in_community=None, block_offer=None, is_chp=None, chp_pars=None,
                 default_alpha=10.0):
        """
        :param settings: a MarketSettings object. contains the time horizon that is needed here.
        :param agent_ids: an array with agents names, should be strings
        :param a_type: array of strings. should be one of "producer, "consumer", "prosumer"
        :param gmax: array of size (nr_of_timesteps, nr_of_agents)
        :param lmax: array of size (nr_of_timesteps, nr_of_agents)
        :param cost:
        :param util:
        :param co2: optional input. array of size (nr_of_timesteps, nr_of_agents)
        :param is_in_community: optional input. Boolean array of size (1, nr_of_agents).
                    contains True if is in community, False if not.
        :param block_offer: # TODO sergio can you define this variable?
        :param is_chp: list with ids of agents that are CHPs
        :params chp_pars: a dictionary of dictionaries, including parameters for each agents in is_chp.
                                                        {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                         'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
        """

        # set nr of agents, names, and types
        self.nr_of_agents = len(agent_ids)
        self.agent_name = agent_ids
        #self.agent_type = dict(zip(name, a_type))
        # add community info if that is needed
        if settings.market_design == "community":
            if is_in_community is None:
                raise ValueError("The community market design is selected. In this case, is_in_community is "
                                 "an obligatory input")
            self.agent_is_in_community = pd.DataFrame(np.reshape(is_in_community, (1, self.nr_of_agents)),
                                                      columns=agent_ids)
            self.C = [i for i in range(self.nr_of_agents) if is_in_community[i]]
            self.notC = [i for i in range(self.nr_of_agents) if not is_in_community[i]]
        else:
            self.agent_is_in_community = None
            self.C = None
            self.notC = None

        # add co2 emission info if needed
        if settings.product_diff == "co2Emissions":
            self.co2_emission = pd.DataFrame(np.reshape(co2, (1, self.nr_of_agents)),
                                             columns=agent_ids)  # 1xnr_of_agents dimension
        else:
            self.co2_emission = None  # pd.DataFrame(np.ones((1, self.nr_of_agents))*np.nan, columns=agent_ids)

        if settings.offer_type == 'block':
            self.block = block_offer
        else:
            self.block = None

        # time dependent data -------------------------------------------------
        if settings.nr_of_h == 1:
            lmin = np.zeros((1, self.nr_of_agents))
            gmin = np.zeros((1, self.nr_of_agents))
            lmax = np.reshape(lmax, (1, self.nr_of_agents))
            gmax = np.reshape(gmax, (1, self.nr_of_agents))
            cost = np.reshape(cost, (1, self.nr_of_agents))
            util = np.reshape(util, (1, self.nr_of_agents))
        else:
            # set lmin and gmin to zero.
            lmin = np.zeros((settings.nr_of_h, self.nr_of_agents))
            gmin = np.zeros((settings.nr_of_h, self.nr_of_agents))
        # check size of inputs
        if not np.array(lmin).shape == (settings.nr_of_h, self.nr_of_agents):
            raise ValueError("lmin has to have shape (nr_of_timesteps, nr_of_agents)")
        # TODO check that prodcers have lmax = 0, consumers have gmax = 0 for all times, min smaller than max, etc.
        self.gmin = pd.DataFrame(gmin, columns=agent_ids)
        self.gmax = pd.DataFrame(gmax, columns=agent_ids)
        self.lmin = pd.DataFrame(lmin, columns=agent_ids)
        self.lmax = pd.DataFrame(lmax, columns=agent_ids)

        self.cost = pd.DataFrame(cost, columns=agent_ids)
        self.util = pd.DataFrame(util, columns=agent_ids)

        # change the self.cost for agents in is_chp if el_dependent option is True
        if settings.el_dependent:
            if is_chp is None:
                raise ValueError("if el_dependent is chosen, the input is_chp must be given")
            if chp_pars is None:
                raise ValueError("if el_dependent is chosen, the input chp_pars must be given")
            # make sure that cph_params.keys() is a subset of is_chp
            if not set(list(chp_pars.keys())).issubset(is_chp):
                raise ValueError("some keys in chp_pars do not belong to the set is_chp")
            # organize the CHP parameters in a dataframe
            self.chp_params = pd.DataFrame.from_dict(chp_pars)
            # defaults for CHP:
            defaults = pd.DataFrame({"col": {"alpha": default_alpha, "r": 0.45, "rho_H": 0.9, "rho_E": 0.25}})
            for i in range(len(is_chp)):
                if is_chp[i] not in chp_pars.keys():
                    self.chp_params[is_chp[i]] = defaults
            # compute the hourly cost bid for each chp
            for i in range(len(is_chp)):
                criterion = self.chp_params.loc["alpha", is_chp[i]] * self.chp_params.loc["rho_E", is_chp[i]]
                for t in range(settings.nr_of_h):
                    if settings.elPrice.iloc[t, 0] <= criterion:
                        self.cost.loc[t, is_chp[i]] = self.chp_params.loc["alpha", is_chp[i]] * (
                                    self.chp_params.loc["rho_E", is_chp[i]] * self.chp_params.loc["r", is_chp[i]] +
                                    self.chp_params.loc["rho_H", is_chp[i]]) - settings.elPrice.iloc[t, 0] * \
                                                      self.chp_params.loc["r", is_chp[i]]
                    else:
                        self.cost.loc[t, is_chp[i]] = settings.elPrice.iloc[t, 0] * (
                                self.chp_params.loc["rho_H", is_chp[i]] / self.chp_params.loc["rho_E", is_chp[i]])
        else:
            self.chp_params = None


# network data ---------------------------------------------------------------------------------------------------------
class Network:
    def __init__(self, agent_data, gis_data, settings, nodes, edges):  # agent_loc,
        """
        :param agent_data: AgentData object.
        :param agent_loc: dictionary mapping agent ids to node numbers
        :param gis_data: dataframe provided by GIS to us. has columns from_to (tuple), losses_total, length, total_costs
        :param settings: a MarketSettings object
        :param nodes: a list of node IDs. IDs of nodes where an agent is located are equal to agent ID
        :param edges: a dataframe including (from, to, installed_capacity  pipe_length surface_type 
                             total_costs  diameter  losses_w_m   losses_w capacity_limit) for each edge
        :output: a Network object with 2 properties: distance and losses (both n by n np.array). distance[1,3] is the
        distance from agent 1 to agent 3. has np.inf if cannot reach the agent.
        """

        if settings.network_type is not None and gis_data is None:
            raise ValueError(
                "gis_data has to be given if network_type is not None"
            )
        if settings.market_design == "p2p" and settings.product_diff != "noPref" and gis_data is None:
            raise ValueError(
                "gis_data has to be given for p2p market with preferences"
            )

        if settings.network_type is not None:
            self.N = nodes
            self.nr_of_n = len(self.N)
            self.P = edges #[(edges.loc[i, "from"], edges.loc[i, "to"]) for i in range(len(edges))]
            self.nr_of_p = len(self.P)
            # make the A matrix
            A = np.zeros((len(self.N), len(self.P)))
            for p_nr in range(self.nr_of_p):
                p = self.P[p_nr]
                n1_nr = np.where(self.N == p[0])
                n2_nr = np.where(self.N == p[1])
                A[n1_nr, p_nr] = 1
                A[n2_nr, p_nr] = -1
            self.A = A
            # define location where agents are
            self.loc_a = agent_data.agent_name # agents are located at the nodes with their own name

        # define distance and losses between any two agents in a matrix ----------------------------
        if settings.market_design == "p2p" and settings.product_diff != "noPref":
            self.distance = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
            self.losses = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
            for i in range(agent_data.nr_of_agents):
                self.distance[i, i] = 0.0  # distance to self is zero.
                self.losses[i, i] = 0.0  # losses to self is zero.
            for row_nr in range(len(gis_data["from_to"].values)):
                (From, To) = gis_data["from_to"].values[row_nr]
                self.distance[From, To] = gis_data.length.iloc[row_nr]
                self.losses[From, To] = gis_data["losses_total"].iloc[row_nr]

            # graph for the Dijkstra's
            graph = {i: {j: np.inf for j in range(0, agent_data.nr_of_agents)} for i in
                    range(0, agent_data.nr_of_agents)}
            total_dist = []  # total network distance

            for j in range(0, agent_data.nr_of_agents):
                for i in range(0, agent_data.nr_of_agents):
                    if self.distance[i][j] != 0 and self.distance[i][j] != np.inf:
                        # symmetric matrix
                        graph[i][j] = self.distance[i][j]
                        graph[j][i] = self.distance[i][j]
                        total_dist.append(self.distance[i][j])

            # Matrix with the distance between all the agents
            self.all_distance = np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))  # might need this later
            for i in range(0, agent_data.nr_of_agents):
                aux = []
                aux = self.calculate_distances(graph, i)
                for j in range(0, agent_data.nr_of_agents):
                    self.all_distance[i][j] = aux[j]
            # network usage in percentage for each trade Pnm
            self.all_distance_percentage = self.all_distance / sum(total_dist)

            # LOSSES
            # graph for the Dijkstra's
            graph = {i: {j: np.inf for j in range(0, agent_data.nr_of_agents)} for i in
                    range(0, agent_data.nr_of_agents)}
            total_losses = []  # total network losses

            for j in range(0, agent_data.nr_of_agents):
                for i in range(0, agent_data.nr_of_agents):
                    if self.losses[i][j] != 0 and self.losses[i][j] != np.inf:
                        # symmetric matrix
                        graph[i][j] = self.losses[i][j]
                        graph[j][i] = self.losses[i][j]
                        total_losses.append(self.losses[i][j])

            # Matrix with the losses between all the agents
            self.all_losses = np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))  # might need this later
            for i in range(0, agent_data.nr_of_agents):
                aux = []
                aux = self.calculate_distances(graph, i)  # calculating losses shortest path
                for j in range(0, agent_data.nr_of_agents):
                    self.all_losses[i][j] = aux[j]
            # network usage in percentage for each trade Pnm
            self.all_losses_percentage = self.all_losses / sum(total_losses)
    
    # DISTANCE
    # Dijkstra's shortest path
    def calculate_distances(self, graph, starting_vertex):
        distances = {vertex: float('infinity') for vertex in graph}
        distances[starting_vertex] = 0

        pq = [(0, starting_vertex)]
        while len(pq) > 0:
            current_distance, current_vertex = heapq.heappop(pq)

            # Nodes can get added to the priority queue multiple times. We only
            # process a vertex the first time we remove it from the priority queue.
            if current_distance > distances[current_vertex]:
                continue
            for neighbor, weight in graph[current_vertex].items():
                distance = current_distance + weight

                # Only consider this new path if it's better than any path we've
                # already found.
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heapq.heappush(pq, (distance, neighbor))
        return distances
