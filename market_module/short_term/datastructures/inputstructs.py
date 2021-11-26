# inputs format for market module
# if we receive them differently from other modules, we will convert them to these

import pandas as pd
import numpy as np
import heapq


# general settings object ---------------------------------------------------------------------------------------------
class MarketSettings:
    """
    Object to store market settings.
    On creation, it checks whether settings are valid
    """

    def __init__(self, nr_of_hours, offer_type: str, prod_diff: str,
                 market_design: str, network_type=None, el_dependent=False, el_price=None):
        """
        create MarketSettings object if all inputs are correct
        :param nr_of_hours: Integer between 1 and ... ?
        :param offer_type:
        :param prod_diff:
        TODO explain all inputs
        """

        max_time_steps = 48  # max 48 hours.
        if not ((type(nr_of_hours) == int) and (1 <= nr_of_hours <= max_time_steps)):
            raise ValueError("nr_of_hours should be an integer between 1 and " + str(max_time_steps))
        self.nr_of_h = nr_of_hours
        self.timestamps = np.arange(nr_of_hours)
        # check if input is correct.
        options_offer_type = ["simple", "block", "energyBudget"]
        if offer_type not in options_offer_type:
            raise ValueError("offer_type should be one of ['simple', 'block', 'energyBudget']")
        self.offer_type = offer_type
        # check if input is correct
        options_prod_diff = ["noPref", "co2Emissions", "networkDistance", "losses"]
        if prod_diff not in options_prod_diff:
            raise ValueError('prod_diff should be one of ' + str(options_prod_diff))
        self.product_diff = prod_diff
        # check if input is correct
        options_market_design = ["pool", "p2p", "community"]
        if market_design not in options_market_design:
            raise ValueError('market_design should be one of ' + str(options_market_design))
        self.market_design = market_design
        # exclude bad combination of inputs
        if self.market_design != "p2p" and prod_diff != "noPref":
            raise ValueError('prod_diff can only be something else than "noPref" if market_design == "p2p')
        # check inputs for electricity dependence. Can be combined with all 3 market types
        self.el_dependent = el_dependent
        if el_dependent:
            if el_price is None:
                raise ValueError('el_price must be given if el_dependent == True')
            elif not len(el_price) == nr_of_hours:
                raise ValueError('el_price must be given for each hour')
            self.elPrice = pd.DataFrame(el_price, columns=["elprice"])
        else:
            self.elPrice = None

        # entries to be filled in by other functions
        self.community_objective = None
        self.gamma_peak = None
        self.gamma_imp = None
        self.gamma_exp = None

        # check network type settings
        if network_type is not None:
            options_network_type = ["direction"]
            if network_type not in options_network_type:
                raise ValueError("network_type should be None or one of " + str(options_network_type))
            if not offer_type == "simple":
                raise ValueError("If you want network-awareness, offer_type must be 'simple'")
            if not market_design == "pool":
                raise NotImplementedError("network-aware is not implemented for p2p and community markets")
        # save network type
        self.network_type = network_type

    def add_community_settings(self, objective, g_peak=10.0 ** 2, g_exp=-4 * 10.0 ** 1, g_imp=5 * 10.0 ** 1):
        """ the parameters are optional inputs"""
        # add the options for community to the settings
        options_objective = ["autonomy", "peakShaving"]
        if objective not in options_objective:
            raise ValueError("objective should be one of" + str(options_objective))
        self.community_objective = objective

        # for now, set default values of gammas
        self.gamma_peak = g_peak
        self.gamma_exp = g_exp
        self.gamma_imp = g_imp
        if self.gamma_exp >= 0.0:
            raise ValueError("export penalty must be nonpositive")


# agents information --------------------------------------------------------------------------------------------------
class AgentData:
    """
    Object that stores all agent related data
    Contains an entry for each different input.
    If the input is constant in time, it is a dataframe with agent ID as column name, and the input as row
    If the input is varying in time, it is a dataframe with agent ID as column name, and time along the rows
    """

    def __init__(self, settings, name, a_type, gmin, gmax, lmin, lmax, cost, util, co2=None,
                 is_in_community=None, block_offer=None, is_chp=None, chp_pars=None,
                 default_alpha=10.0):
        """
        :param settings: a MarketSettings object. contains the time horizon that is needed here.
        :param name: an array with agents names, should be strings
        :param a_type: array of strings. should be one of "producer, "consumer", "prosumer"
        :param gmin: array of size (nr_of_timesteps, nr_of_agents)
        :param gmax: array of size (nr_of_timesteps, nr_of_agents)
        :param lmin: array of size (nr_of_timesteps, nr_of_agents)
        :param lmax: array of size (nr_of_timesteps, nr_of_agents)
        :param cost:
        :param util:
        :param co2: optional input. array of size (nr_of_timesteps, nr_of_agents)
        :param is_in_community: optional input. Boolean array of size (1, nr_of_agents).
                    contains True if is in community, False if not.
        :param block_offer: TODO sergio can you define this variable?
        :param is_chp: list with ids of agents that are CHPs
        :params chp_pars: a dictionary of dictionaries, including parameters for each agents in is_chp.
                                                        {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                         'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
        """

        # set nr of agents, names, and types
        self.nr_of_agents = len(name)
        self.agent_name = name
        self.agent_type = dict(zip(name, a_type))
        # add community info if that is needed
        if settings.market_design == "community":
            if is_in_community is None:
                raise ValueError("The community market design is selected. In this case, is_in_community is "
                                 "an obligatory input")
            self.agent_is_in_community = pd.DataFrame(np.reshape(is_in_community, (1, self.nr_of_agents)),
                                                      columns=name)
            self.C = [i for i in range(self.nr_of_agents) if is_in_community[i]]
            self.notC = [i for i in range(self.nr_of_agents) if not is_in_community[i]]
        else:
            self.agent_is_in_community = None
            self.C = None
            self.notC = None

        # add co2 emission info if needed
        if settings.product_diff == "co2Emissions":
            self.co2_emission = pd.DataFrame(np.reshape(co2, (1, self.nr_of_agents)),
                                             columns=name)  # 1xnr_of_agents dimension
        else:
            self.co2_emission = None  # pd.DataFrame(np.ones((1, self.nr_of_agents))*np.nan, columns=name)

        if settings.offer_type == 'block':
            self.block = block_offer

        # time dependent data -------------------------------------------------
        if settings.nr_of_h == 1:
            lmin = np.reshape(lmin, (1, self.nr_of_agents))
            gmin = np.reshape(gmin, (1, self.nr_of_agents))
            lmax = np.reshape(lmax, (1, self.nr_of_agents))
            gmax = np.reshape(gmax, (1, self.nr_of_agents))
            cost = np.reshape(cost, (1, self.nr_of_agents))
            util = np.reshape(util, (1, self.nr_of_agents))
        # check size of inputs
        if not np.array(lmin).shape == (settings.nr_of_h, self.nr_of_agents):
            raise ValueError("lmin has to have shape (nr_of_timesteps, nr_of_agents)")
        # TODO check that prodcers have lmax = 0, consumers have gmax = 0 for all times, min smaller than max, etc.
        self.gmin = pd.DataFrame(gmin, columns=name)
        self.gmax = pd.DataFrame(gmax, columns=name)
        self.lmin = pd.DataFrame(lmin, columns=name)
        self.lmax = pd.DataFrame(lmax, columns=name)

        self.cost = pd.DataFrame(cost, columns=name)
        self.util = pd.DataFrame(util, columns=name)

        # change the self.cost for agents in is_chp if el_dependent option is True
        if settings.el_dependent:
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
    def __init__(self, agent_data, gis_data, settings):  # agent_loc,
        """
        :param agent_data: AgentData object.
        :param agent_loc: dictionary mapping agent ids to node numbers
        :param gis_data: dataframe provided by GIS to us. has columns from/to (tuple), Losses total (W), length (m),
                        total costs
        :param settings: a MarketSettings object
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
            # extract node numbers from GIS data
            nodes = np.array(list(set([item for t in gis_data["From/to"] for item in t])))
            self.N = nodes
            self.nr_of_n = len(self.N)
            self.P = gis_data["From/to"]  # tuples
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
            self.loc_a = self.N  # TODO for now, put this. can be removed later

        # define distance and losses between any two agents in a matrix ----------------------------
        if settings.market_design == "p2p" and settings.product_diff != "noPref":
            self.distance = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
            self.losses = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
            for i in range(agent_data.nr_of_agents):
                self.distance[i, i] = 0.0  # distance to self is zero.
                self.losses[i, i] = 0.0  # losses to self is zero.
            for row_nr in range(len(gis_data["From/to"].values)):
                (From, To) = gis_data["From/to"].values[row_nr]
                self.distance[From, To] = gis_data.Length.iloc[row_nr]
                self.losses[From, To] = gis_data["Losses total [W]"].iloc[row_nr]

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
