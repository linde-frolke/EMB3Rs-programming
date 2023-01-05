# inputs format for market module

import pandas as pd
import numpy as np
import heapq
from typing import List, Any, Union, Dict
from pydantic import BaseModel, validator, Field, constr, conint

# from market_module.cases.exceptions.module_validation_exception import ModuleValidationException #, confloat

# general settings object ---------------------------------------------------------------------------------------------
class MarketSettings(BaseModel):
    """
    Object to store market settings.
    On creation, it checks whether settings are valid
    """
    class Config:
        arbitrary_types_allowed = True
    
    nr_of_h : conint(strict=True)  # nr of time steps to run the market for
    market_design : constr(strict=True) # market design
    offer_type : constr(strict=True) # offer type is either "simple", "block", "energyBudget".
    product_diff : constr(strict=True) = "noPref" # product differentiation. Only has an effect if market design = p2p
    el_price : Union[None, np.ndarray] = Field(default=None) # a vector with electricity prices. is None by default. 
    el_dependent : bool # should CHP bids be depending on electricity price?
    network_type : str = None  # whether to include network, and if so, how. Default is None
    # entries to be filled in by other functions
    # filled in by init
    timestamps : Any = Field(default=None)
    elPrice : Any = Field(default=None)
    gamma_peak : Union[None, float]
    community_objective : Union[None, str] # constr(regex=r'(autonomy|peakShaving)$')]
    gamma_exp : Union[None, float]
    gamma_imp : Union[None, float]
    solver: str

    def __init__(self, **data) -> None:
        """
        create MarketSettings object if all inputs are correct
        """
        # pydantic __init__ syntax
        super().__init__(**data)
        
        # here we initiate the entries that are computed from inputs
        self.timestamps = np.arange(self.nr_of_h)

        if self.el_dependent:
            self.elPrice = pd.DataFrame(self.el_price, columns=["elprice"])
        else:
            self.elPrice = None

    @validator("nr_of_h")
    def nr_of_hours_check(cls, v):
        max_time_steps = 24*366  # max 1 year   
        if not ((type(v) == int) and (1 <= v <= max_time_steps)):
            raise ValueError("nr_of_hours should be an integer between 1 and " + str(max_time_steps))
        return v
    @validator("offer_type")
    def offer_type_valid(cls, v):
        options_offer_type = ["simple", "block", "energyBudget"]
        if v not in options_offer_type:
            raise ValueError("offer_type should be one of " + str(options_offer_type))
        return v
    def no_block_with_community(cls, v, values):
        if (v == "block" and values["market_design"] == "community"):
            raise ValueError("Block bids cannot be combined with community market. \n " + 
                            "Choose a different market design or set bid format to 'simple' or 'energyBudget'.")
        return v
    @validator("product_diff")
    def prof_diff_valid(cls, v):
        options_product_diff = ["noPref", "co2Emissions", "networkDistance", "losses"]
        if v not in options_product_diff:
            raise ValueError('product_diff should be one of ' + str(options_product_diff))
        return v
    @validator("product_diff")
    def product_diff_only_with_p2p(cls, v, values):
        # exclude bad combination of inputs
        try:
            if values["market_design"] != "p2p" and v != "noPref":
                raise ValueError('product_diff can only be something else than "noPref" if market_design == "p2p')
        except:
            raise ValueError("market design input was invalid, I cannot run this test")
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
            elif not len(values["el_price"]) == values["nr_of_h"]:
                raise ValueError('el_price must be given for each hour')
        return v
    @validator("network_type")
    def network_type_valid(cls, v, values):
        if v is not None:
            options_network_type = ["direction"]
            if v not in options_network_type:
                raise ValueError("network_type should be None or one of " + str(options_network_type))
            if values["offer_type"] == "block":
                raise ValueError("If you want network-awareness, offer_type cannot be 'block'. \n" + 
                "Please choose 'simple' or 'energyBudget' instead. ")
            if not values["market_design"] == "pool":
                raise ValueError("network-awareness is only implemented for pool, not for p2p and community markets")
        return v
    @validator("community_objective")
    def community_inputs_must_be_given(cls, v, values):
        if values["market_design"] == "community" and v is None:
            raise ValueError("Community_objective is mandatory input if the community market design is selected")
        return v 
    @validator("community_objective")
    def community_objective_valid(cls, v, values):
        options_objective = ["autonomy", "peakShaving"]
        if values["market_design"] == "community" and v not in options_objective:
            raise ValueError("community objective should be one of" + str(options_objective))
        return v
    @validator("gamma_exp")
    def gamma_exp_validity_check(cls, v, values):
        if values["market_design"] == "community" and values["community_objective"] == "autonomy":
            if v is None:
                raise ValueError("gamma_exp is mandatory input if the autonomy objective for community market is selected")
            else:
                if v < 0:
                    raise ValueError("gamma_exp must be positive")
        return v
    @validator("gamma_imp")
    def gamma_imp_validity_check(cls, v, values):
        if values["market_design"] == "community" and values["community_objective"] == "autonomy":
            if v is None:
                raise ValueError("gamma_imp is mandatory input if the autonomy objective for community market is selected")
            else:
                if v < 0:
                    raise ValueError("gamma_imp must be positive")
                else:
                    if abs(v) < abs(values["gamma_exp"]):
                        raise ValueError("In absolute value, gamma_imp should be greater than gamma_exp")
        return v
    @validator("community_objective")
    def g_peak_validity_check(cls, v, values):
        if values["market_design"] == "community" and v == "peakShaving":
            if values["gamma_peak"] is None:
                raise ValueError("g_peak is mandatory input if the community market design" +\
                                 "with peak shaving objective is selected")
            else:
                if values["gamma_peak"]  < 0:
                    raise ValueError("g_peak must be positive")
        return v
    @validator("solver")
    def solver_implemented(cls, v):
        if v not in ["SCIP", "GUROBI", "HIGHS", "COPT"]:
            raise ValueError("solver should be SCIP, GUROBI, HIGHS or COPT")
        return v
    
        

# agents information --------------------------------------------------------------------------------------------------
class AgentData(BaseModel):
    """
    Object that stores all agent related data/inputs needed for the market
    :param settings: a MarketSettings object. contains the time horizon that is needed here.
    :param agent_name: an array with agents names, should be strings
    :param gmax: array of size (nr_of_timesteps, nr_of_agents)
    :param lmax: array of size (nr_of_timesteps, nr_of_agents)
    :param cost:
    :param util:
    :param co2: optional input. array of size (nr_of_timesteps, nr_of_agents)
    :param is_in_community: optional input. Boolean array of size (1, nr_of_agents).
                contains True if is in community, False if not.
    :param block_offer: # 
    :param is_chp: list with ids of agents that are CHPs
    :params chp_pars: a dictionary of dictionaries, including parameters for each agents in is_chp.
                                                    {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                        'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
    """
    # only needed for constructing validation
    settings : MarketSettings
    # real needed entries
    agent_name : List
    gmax : List 
    lmax : List
    cost : List 
    util : List 
    co2 : Union[None, List] = None 
    is_in_community : Union[None, List] = None
    block : Union[None, Dict] = None
    is_chp : Union[None, List] = None # list of agent names 
    chp_pars : Union[None, Dict] = None
    default_alpha : float = 10.0
    # to be filled in init.
    co2_emission : Any
    nr_of_agents : Any
    agent_is_in_community : Any 
    C : Any
    notC : Any
    gmin : Any
    lmin : Any  

    def __init__(self, **data) -> None:
        # pydantic __init__ syntax
        super().__init__(**data)

        # set nr of agents, names, and types
        self.nr_of_agents = len(self.agent_name)
        
        # add community info if that is needed
        if self.settings.market_design == "community":
            self.agent_is_in_community = pd.DataFrame(np.reshape(self.is_in_community, (1, self.nr_of_agents)),
                                                      columns=self.agent_name)
            self.C = [i for i in range(self.nr_of_agents) if self.is_in_community[i]]
            self.notC = [i for i in range(self.nr_of_agents) if not self.is_in_community[i]]
        else:
            self.agent_is_in_community = None
            self.C = None
            self.notC = None

        # add co2 emission info if needed
        if self.settings.product_diff == "co2Emissions":
            self.co2_emission = pd.DataFrame(np.reshape(self.co2, (1, self.nr_of_agents)),
                                             columns=self.agent_name)  # 1xnr_of_agents dimension

        # time dependent data -------------------------------------------------
        if self.settings.nr_of_h == 1:
            self.lmin = np.zeros((1, self.nr_of_agents))
            self.gmin = np.zeros((1, self.nr_of_agents))
            self.lmax = np.reshape(self.lmax, (1, self.nr_of_agents))
            self.gmax = np.reshape(self.gmax, (1, self.nr_of_agents))
            self.cost = np.reshape(self.cost, (1, self.nr_of_agents))
            self.util = np.reshape(self.util, (1, self.nr_of_agents))
        else:
            # set lmin and gmin to zero.
            self.lmin = np.zeros((self.settings.nr_of_h, self.nr_of_agents))
            self.gmin = np.zeros((self.settings.nr_of_h, self.nr_of_agents))
        
        # create the dataframes
        self.gmin = pd.DataFrame(self.gmin, columns=self.agent_name)
        self.gmax = pd.DataFrame(self.gmax, columns=self.agent_name)
        self.lmin = pd.DataFrame(self.lmin, columns=self.agent_name)
        self.lmax = pd.DataFrame(self.lmax, columns=self.agent_name)

        self.cost = pd.DataFrame(self.cost, columns=self.agent_name)
        self.util = pd.DataFrame(self.util, columns=self.agent_name)

        # change the self.cost for agents in is_chp if el_dependent option is True
        if self.settings.el_dependent:
            # organize the CHP parameters in a dataframe
            chp_params = pd.DataFrame.from_dict(self.chp_pars)
            # set defaults for CHP:
            defaults = pd.DataFrame({"col": {"alpha": self.default_alpha, "r": 0.45, "rho_H": 0.9, "rho_E": 0.25}})
            for i in range(len(self.is_chp)):
                if self.is_chp[i] not in self.chp_pars.keys():
                    chp_params[self.is_chp[i]] = defaults
            # replace it with the new
            self.chp_pars = chp_params
            
            # compute the hourly cost bid for each chp
            for i in range(len(self.is_chp)):
                criterion = chp_params.loc["alpha", self.is_chp[i]] * chp_params.loc["rho_E", self.is_chp[i]]
                for t in range(self.settings.nr_of_h):
                    if self.settings.elPrice.iloc[t, 0] <= criterion:
                        self.cost.loc[t, self.is_chp[i]] = chp_params.loc["alpha", self.is_chp[i]] * (
                                    chp_params.loc["rho_E", self.is_chp[i]] * chp_params.loc["r", self.is_chp[i]] +
                                    chp_params.loc["rho_H", self.is_chp[i]]) - self.settings.elPrice.iloc[t, 0] * \
                                                      chp_params.loc["r", self.is_chp[i]]
                    else:
                        self.cost.loc[t, self.is_chp[i]] = self.settings.elPrice.iloc[t, 0] * (
                                chp_params.loc["rho_H", self.is_chp[i]] / chp_params.loc["rho_E", self.is_chp[i]])
    @validator("agent_name")
    def agent_ids_unique(cls, v):
        if not len(set(v)) == len(v):
            raise ValueError("agent_name must contain unique agent ids -- no two ids may be the same")
        return v
    @validator("gmax")
    def gmax_nrofh_lists(cls, v, values):
        if len(v) != values["settings"].nr_of_h:
            raise ValueError("gmax should be a list of nr_of_h=" + str(values["settings"].nr_of_h) + " lists. ")
        return v
    @validator("gmax")
    def gmax_nrofagents_lists(cls, v, values):
        sublist_length_correct = [len(i) == len(values["agent_name"]) for i in v]
        if not all(sublist_length_correct):
            raise ValueError("Each sublist in gmax sould be of length nr_of_agents=" + str(len(values["agent_name"])))
        return v
    @validator("lmax")
    def lmax_nrofh_lists(cls, v, values):
        if len(v) != values["settings"].nr_of_h:
            raise ValueError("lmax should be a list of nr_of_h=" + str(values["settings"].nr_of_h) + " lists. ")
        return v
    @validator("lmax")
    def lmax_nrofagents_lists(cls, v, values):
        sublist_length_correct = [len(i) == len(values["agent_name"]) for i in v]
        if not all(sublist_length_correct):
            raise ValueError("Each sublist in lmax sould be of length nr_of_agents=" + str(len(values["agent_name"])))
        return v
    @validator("cost")
    def cost_nrofh_lists(cls, v, values):
        if len(v) != values["settings"].nr_of_h:
            raise ValueError("cost should be a list of nr_of_h=" + str(values["settings"].nr_of_h) + " lists. ")
        return v
    @validator("cost")
    def cost_nrofagents_lists(cls, v, values):
        sublist_length_correct = [len(i) == len(values["agent_name"]) for i in v]
        if not all(sublist_length_correct):
            raise ValueError("Each sublist in cost sould be of length nr_of_agents=" + str(len(values["agent_name"])))
        return v
    @validator("cost")
    def cost_nonnegative(cls, v):
        lowest_cost = min(min(v))
        if lowest_cost < 0:
            raise ValueError("The cost bids must be nonnegative")
        return v
    @validator("util")
    def util_nrofh_lists(cls, v, values):
        if len(v) != values["settings"].nr_of_h:
            raise ValueError("util should be a list of nr_of_h=" + str(values["settings"].nr_of_h) + " lists. ")
        return v
    @validator("util")
    def util_nrofagents_lists(cls, v, values):
        sublist_length_correct = [len(i) == len(values["agent_name"]) for i in v]
        if not all(sublist_length_correct):
            raise ValueError("Each sublist in util sould be of length nr_of_agents=" + str(len(values["agent_name"])))
        return v
    @validator("util")
    def util_nonnegative(cls, v):
        lowest_util = min(min(v))
        if lowest_util < 0:
            raise ValueError("The utility bids must be nonnegative")
        return v
    @validator("is_in_community")
    def community_parameters_given(cls, v, values):
        if v is None and values["settings"].market_design == "community":
            raise ValueError("If the community market design is selected, is_in_community is "
                                 "an obligatory input")
        return v
    @validator("co2")
    def co2_given_if_needed(cls, v, values):
        if values["settings"].product_diff == "co2Emissions":
            if v is None:
                raise ValueError("co2 intensity for agents is a mandatory input since you selected" +\
                                    "product_diff = co2Emissions")
            else:
                if not len(v) == len(values["agent_name"]):
                    raise ValueError("'co2' has to be a list of size nr_of_agents=" +\
                         str(len(values["agent_name"])))
        return v
    @validator("is_chp")
    def is_chp_given_and_correct_if_needed(cls, v, values):
        if values["settings"].el_dependent:
            if v is None:
                raise ValueError("if el_dependent is chosen, the input is_chp is mandatory")
            elif not all([v[i] in values["agent_name"] for i in range(len(v))]):
                raise ValueError("the strings in 'is_chp' have to be present in 'agent_names'")
        return v
    @validator("chp_pars")
    def chp_pars_given_if_needed(cls, v, values):
        if v is None and values["settings"].el_dependent:
            raise ValueError("if el_dependent is chosen, the input chp_pars is mandatory")
        return v
    @validator("chp_pars")
    def chp_pars_keys_must_be_chps(cls, v, values):
        # make sure that cph_params.keys() is a subset of is_chp
        if values["settings"].el_dependent:
            if not set(list(v.keys())).issubset(values["is_chp"]):
                raise ValueError("some keys in chp_pars do not belong to the set is_chp")
        return v


# network data ---------------------------------------------------------------------------------------------------------
class Network(BaseModel):
    """
        :param agent_data: AgentData object.
        :param agent_loc: dictionary mapping agent ids to node numbers
        :param gis_data: dataframe provided by GIS to us. has columns from_to (tuple), losses_total, length, total_costs
        :param settings: a MarketSettings object
        :param N: a list of node IDs. IDs of nodes where an agent is located are equal to agent ID
        :param P: a dataframe including (from, to, installed_capacity  pipe_length surface_type 
                             total_costs  diameter  losses_w_m   losses_w capacity_limit) for each edge
        :output: a Network object with 2 properties: distance and losses (both n by n np.array). distance[1,3] is the
        distance from agent 1 to agent 3. has np.inf if cannot reach the agent.
        """
    class Config:
        arbitrary_types_allowed = True
    # 
    settings : MarketSettings
    agent_data : AgentData
    gis_data : Union[None, pd.DataFrame]
    N : Union[List, None]
    P : Union[List, None]
    # to fill for network
    nr_of_n : Any = None
    nr_of_p : Any = None
    A : Any = None 
    loc_a : Any= None
    # to fill for p2p
    all_distance_percentage : Any = None
    all_losses_percentage : Any = None
    emissions_percentage : Any = None

    def __init__(self, **data) -> None:
        # pydantic __init__ syntax
        super().__init__(**data)

        if self.settings.network_type is not None:  
            self.nr_of_n = len(self.N)
            self.nr_of_p = len(self.P)
            # make the A matrix
            A = np.zeros((len(self.N), len(self.P)))
            for p_nr in range(self.nr_of_p):
                p = self.P[p_nr]
                n1_nr = self.N.index(p[0])
                n2_nr = self.N.index(p[1])
                A[n1_nr, p_nr] = 1
                A[n2_nr, p_nr] = -1
            self.A = A
            # define location where agents are
            self.loc_a = self.agent_data.agent_name # agents are located at the nodes with their own name

        # define distance and losses between any two agents in a matrix ----------------------------
        if self.settings.market_design == "p2p":
            # emissions percentage:
            if self.settings.product_diff == "co2Emissions":
                self.emissions_percentage = self.agent_data.co2_emission / sum(self.agent_data.co2_emission.T[0])  # percentage
            # if preferences based on losses or networkdistance is selected, we need to compute those:
            elif self.settings.product_diff in ["losses", "networkDistance"]:
                distance = np.inf * np.ones((self.agent_data.nr_of_agents, self.agent_data.nr_of_agents))
                losses = np.inf * np.ones((self.agent_data.nr_of_agents, self.agent_data.nr_of_agents))
                for i in range(self.agent_data.nr_of_agents):
                    distance[i, i] = 0.0  # distance to self is zero.
                    losses[i, i] = 0.0  # losses to self is zero.
                for row_nr in range(len(self.gis_data["from_to"].values)):
                    (From, To) = self.gis_data["from_to"].values[row_nr]
                    from_ind = self.agent_data.agent_name.index(From)
                    to_ind = self.agent_data.agent_name.index(To)
                    distance[from_ind, to_ind] = self.gis_data["length"].iloc[row_nr]
                    losses[from_ind, to_ind] = self.gis_data["losses_total"].iloc[row_nr]

                # graph for the Dijkstra's
                graph = {i: {j: np.inf for j in range(0, self.agent_data.nr_of_agents)} for i in
                        range(0, self.agent_data.nr_of_agents)}
                total_dist = []  # total network distance

                for j in range(0, self.agent_data.nr_of_agents):
                    for i in range(0, self.agent_data.nr_of_agents):
                        if distance[i][j] != 0 and distance[i][j] != np.inf:
                            # symmetric matrix
                            graph[i][j] = distance[i][j]
                            graph[j][i] = distance[i][j]
                            total_dist.append(distance[i][j])

                # Matrix with the distance between all the agents
                all_distance = np.ones((self.agent_data.nr_of_agents, self.agent_data.nr_of_agents))  # might need this later
                for i in range(0, self.agent_data.nr_of_agents):
                    aux = []
                    aux = self.calculate_distances(graph, i)
                    for j in range(0, self.agent_data.nr_of_agents):
                        all_distance[i][j] = aux[j]
                # network usage in percentage for each trade Pnm
                self.all_distance_percentage = all_distance / sum(total_dist)

                # LOSSES
                # graph for the Dijkstra's
                graph = {i: {j: np.inf for j in range(0, self.agent_data.nr_of_agents)} for i in
                        range(0, self.agent_data.nr_of_agents)}
                total_losses = []  # total network losses

                for j in range(0, self.agent_data.nr_of_agents):
                    for i in range(0, self.agent_data.nr_of_agents):
                        if losses[i][j] != 0 and losses[i][j] != np.inf:
                            # symmetric matrix
                            graph[i][j] = losses[i][j]
                            graph[j][i] = losses[i][j]
                            total_losses.append(losses[i][j])

                # Matrix with the losses between all the agents
                all_losses = np.ones((self.agent_data.nr_of_agents, self.agent_data.nr_of_agents))  # might need this later
                for i in range(0, self.agent_data.nr_of_agents):
                    aux = []
                    aux = self.calculate_distances(graph, i)  # calculating losses shortest path
                    for j in range(0, self.agent_data.nr_of_agents):
                        all_losses[i][j] = aux[j]
                # network usage in percentage for each trade Pnm
                self.all_losses_percentage = all_losses / sum(total_losses)
            
    
    @validator("N")
    def N_given_if_network(cls, v, values):
        if values["settings"].network_type is not None and v is None:
            raise ValueError("'N' has to be given if network_type is not None")
        return v 
    @validator("P")
    def P_given_if_network(cls, v, values):
        if values["settings"].network_type is not None and v is None:
            raise ValueError("'P' has to be given if network_type is not None")
        return v 
    @validator("gis_data")
    def gis_data_mandatory_if_p2p_and_loss_or_distance(cls, v, values):
        if values["settings"].market_design == "p2p" and (
            values["settings"].product_diff in ["networkDistance", "losses"]):
            if v is None:
                raise ValueError(
                "gis_data has to be given for p2p market with 'networkDistance'- or 'losses'-based preferences"
                )
            else:
                if not set(v.columns) == set(['from_to', 'losses_total', 'length', 'total_costs']):
                    raise ValueError("the column names of gis_data are incorrect. They should be " +\
                        "['from_to', 'losses_total', 'length', 'total_costs']")
                fromto_ids_set = set(np.array([item for t in v.from_to for item in t]))
                if not fromto_ids_set.issubset(set(values["agent_data"].agent_name)):
                    raise ValueError("the tuples in the column gis_data.from_to must only " +\
                        "contain agent names. You entered one or more invalid agent IDs there.")
        return v

    
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
