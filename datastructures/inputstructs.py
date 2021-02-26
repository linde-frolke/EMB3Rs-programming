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
                 market_design: str):
        """
        create MarketSettings object if all inputs are correct
        :param nr_of_hours: Integer between 1 and ... ?
        :param offer_type:
        :param prod_diff:
        """

        max_time_steps = 48         # max 48 hours.
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

        # entries to be filled in by other functions
        self.community_objective = None
        self.gamma_peak = None
        self.gamma_imp = None
        self.gamma_exp = None
        # TODO "integrated with electricity" option
        # TODO ELECTRICITY PRICE HERE

    def add_community_settings(self, objective, g_peak=10.0**2, g_exp=-4 * 10.0**1, g_imp=5 * 10.0**1):
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
                 is_in_community=None):
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
        """

        # TODO print("todo, check all input types")
        # set nr of agents, names, and types
        self.nr_of_agents = len(name)
        # TODO print("todo make sure no ID is identical")
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
        else:
            self.agent_is_in_community = None
        # add co2 emission info if needed
        if settings.product_diff == "co2Emissions":
            self.co2_emission = pd.DataFrame(np.reshape(co2, (1, self.nr_of_agents)), columns=name) #1xnr_of_agents dimension
        else:
            self.co2_emission = None # pd.DataFrame(np.ones((1, self.nr_of_agents))*np.nan, columns=name)

        # time dependent data -------------------------------------------------
        # check size of inputs
        if not lmin.shape == (settings.nr_of_h, self.nr_of_agents):
            raise ValueError("lmin has to have shape (nr_of_timesteps, nr_of_agents)")
        # TODO check that prodcers have lmax = 0, consumers have gmax = 0 for all times
        self.gmin = pd.DataFrame(gmin, columns=name)
        self.gmax = pd.DataFrame(gmax, columns=name)
        self.lmin = pd.DataFrame(lmin, columns=name)
        self.lmax = pd.DataFrame(lmax, columns=name)

        self.cost = pd.DataFrame(cost, columns=name)
        self.util = pd.DataFrame(util, columns=name)


# network data ---------------------------------------------------------------------------------------------------------
class Network:
    def __init__(self, agent_data, gis_data): # agent_loc,
        """
        :param agent_data: AgentData object.
        :param agent_loc: dictionary mapping agent ids to node numbers
        :param gis_data: dataframe provided by GIS to us. has columns from/to (tuple), Losses total (W), length (m), total costs
        :output: a Network object with 2 properties: distance and losses (both n by n np.array). distance[1,3] is the
        distance from agent 1 to agent 3. has np.inf if cannot reach the agent.
        """
        # TODO get this data from GIS module.
        # # node numbers
        # self.N = nodes
        # self.E = edges
        # define location where agents are
        # self.loc_a = agent_loc  # TODO map agent id to node numbers

        # define distance and losses between any two agents in a matrix ----------------------------
        self.distance = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
        self.losses = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
        for i in range(agent_data.nr_of_agents):
            self.distance[i, i] = 0.0   # distance to self is zero.
            self.losses[i, i] = 0.0     # losses to self is zero.
        for row_nr in range(len(gis_data["From/to"].values)):
            (From, To) = gis_data["From/to"].values[row_nr]
            self.distance[From, To] = gis_data.Length.iloc[row_nr]
            self.losses[From, To] = gis_data["Losses total [W]"].iloc[row_nr]

# =============================================================================
# Star here 
# =============================================================================
        #DISTANCE
        #Dijkstra's shortest path        
        def calculate_distances(graph, starting_vertex):
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
        #graph for the Dijkstra's
        graph={i:{j:np.inf for j in range(0,(agent_data.nr_of_agents))} for i in range(0,(agent_data.nr_of_agents))}
        total_dist=[] #total network distance
        
        for j in range(0,(agent_data.nr_of_agents)): 
            for i in range(0,(agent_data.nr_of_agents)):
                if self.distance[i][j]!=0 and self.distance[i][j]!=np.inf:
                    #symmetric matrix
                    graph[i][j]=self.distance[i][j]
                    graph[j][i]=self.distance[i][j]
                    total_dist.append(self.distance[i][j])
         
        #Matrix with the distance between all the agents    
        self.all_distance=np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents)) #might need this later
        for i in range(0,(agent_data.nr_of_agents)):
            aux=[]
            aux=calculate_distances(graph,i)
            for j in range(0,(agent_data.nr_of_agents)):
                self.all_distance[i][j]=aux[j]
        #network usage in percentage for each trade Pnm        
        self.all_distance_percentage=self.all_distance/sum(total_dist)


        #LOSSES
        #graph for the Dijkstra's
        graph={i:{j:np.inf for j in range(0,(agent_data.nr_of_agents))} for i in range(0,(agent_data.nr_of_agents))}
        total_losses=[] #total network losses
        
        for j in range(0,(agent_data.nr_of_agents)): 
            for i in range(0,(agent_data.nr_of_agents)):
                if self.losses[i][j]!=0 and self.losses[i][j]!=np.inf:
                    #symmetric matrix
                    graph[i][j]=self.losses[i][j]
                    graph[j][i]=self.losses[i][j]
                    total_losses.append(self.losses[i][j])
         
        #Matrix with the losses between all the agents    
        self.all_losses=np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents)) #might need this later
        for i in range(0,(agent_data.nr_of_agents)):
            aux=[]
            aux=calculate_distances(graph,i) #calculuting losses shortest path
            for j in range(0,(agent_data.nr_of_agents)):
                self.all_losses[i][j]=aux[j]
        #network usage in percentage for each trade Pnm        
        self.all_losses_percentage=self.all_losses/sum(total_losses)





























