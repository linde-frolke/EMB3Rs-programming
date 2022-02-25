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
    def __init__(self,  prod_diff: str,
                 market_design: str, horizon_b: str, recurr, ydr, data_prof):
        """
        create MarketSettings object if all inputs are correct
        :param offer_type:
        :param prod_diff:
        """

        # check if input is correct
        options_prod_diff = ["noPref", "co2Emissions", "networkDistance"]
        if prod_diff not in options_prod_diff:
            raise ValueError('prod_diff should be one of ' + str(options_prod_diff))
        self.product_diff = prod_diff
        # check if input is correct
        options_market_design = ["centralized", "decentralized"]
        if market_design not in options_market_design:
            raise ValueError('market_design should be one of ' + str(options_market_design))
        self.market_design = market_design
        options_horizon_basis = ['weeks', 'months', 'years']
        if horizon_b not in options_horizon_basis:
            raise ValueError('Horizon basis should be one of ' + str(options_horizon_basis))
        self.horizon_basis = horizon_b
        if not ((type(recurr) == int )):
            raise ValueError("Recurrence should be an integer")
        self.recurrence = recurr
        self.ydr=ydr
        options_data_profile=['hourly', 'daily']
        if data_prof not in options_data_profile:
            raise ValueError('Data profile should be one of ' + str(options_data_profile))
        self.data_profile = data_prof


# agents information --------------------------------------------------------------------------------------------------
class AgentData:
    """
    Object that stores all agent related data
    Contains an entry for each different input.
    If the input is constant in time, it is a dataframe with agent ID as column name, and the input as row
    If the input is varying in time, it is a dataframe with agent ID as column name, and time along the rows
    """
    def __init__(self, settings, name, gmax, lmax, cost, util, co2=None):
        # set nr of agents, names, and types
        self.nr_of_agents = len(name)
        # TODO print("todo make sure no ID is identical")
        self.agent_name = name
        #self.agent_type = dict(zip(name, a_type))
        # add co2 emission info if needed
        if settings.product_diff == "co2Emissions":
            self.co2_emission = pd.DataFrame(np.reshape(co2, (1, self.nr_of_agents)), columns=name) #1xnr_of_agents dimension
        else:
            self.co2_emission = None # pd.DataFrame(np.ones((1, self.nr_of_agents))*np.nan, columns=name)
        #converting horizon_basis to scalar
        if settings.horizon_basis == 'weeks':
            self.day_range = 7
        elif settings.horizon_basis == 'months':
            self.day_range = 30
        else:
            self.day_range = 365   
        
        #converting data profile to scalar
        if settings.data_profile == 'hourly':
            self.data_size = 24
        else:
            self.data_size = 1
       #horizon basis time boundaries
        if settings.horizon_basis == 'weeks' or settings.horizon_basis == 'months':                                     
            if settings.horizon_basis == 'weeks' and settings.recurrence > 52:
                raise ValueError('For horizon basis weeks, recurrence must not exceed 52')
            if settings.horizon_basis == 'months' and settings.recurrence > 12:
                raise ValueError('For horizon basis weeks, recurrence must not exceed 12')

            #These are parameters now

            lmin = np.zeros((self.day_range * settings.recurrence * self.data_size, self.nr_of_agents))
            gmin = np.zeros((self.day_range * settings.recurrence * self.data_size, self.nr_of_agents))
            self.gmin = pd.DataFrame(gmin, columns=name)
            self.lmin = pd.DataFrame(lmin, columns=name)

             #checking data dimensions
            if len(gmin) + len(gmax) + len(lmin) + len(lmax) + len(cost) + len(util) != 6*self.day_range*settings.recurrence*self.data_size:
                raise ValueError('Data dimensions should be {:d}'.format(self.day_range*settings.recurrence*self.data_size))
            

            self.gmax = pd.DataFrame(gmax, columns=name)
            self.lmax = pd.DataFrame(lmax, columns=name)
    
            self.cost = pd.DataFrame(cost, columns=name)
            self.util = pd.DataFrame(util, columns=name)
            
        #Increasing data size if >8760 is required
        elif settings.horizon_basis == 'years':
            if settings.recurrence>20:
                raise ValueError('For horizon basis years, recurrence must not exceed 20')    
            if len(gmin) + len(gmax) + len(lmin) + len(lmax) + len(cost) + len(util) != 6*self.day_range*self.data_size:
                raise ValueError('Data dimensions should be {:d}'.format(self.day_range*self.data_size))
            

            lmax = self.yearly_demand_rate(lmax, settings) #demand increase
            gmax = self.replicate_data(gmax, settings)
            
            cost = self.replicate_data(cost, settings)
            util = self.replicate_data(util, settings)
 
        # time dependent data -------------------------------------------------
            self.gmax = pd.DataFrame(gmax, columns=name)
            self.lmax = pd.DataFrame(lmax, columns=name)
    
            self.cost = pd.DataFrame(cost, columns=name)
            self.util = pd.DataFrame(util, columns=name)
 
# =============================================================================
#     def cumulative_sum(self, nested_list, settings):
#         #defining cumulative sums over weeks, months or years        
#         x=np.zeros([settings.recurrence,self.nr_of_agents])
#         for i in range(0,self.nr_of_agents):
#             for k,j in enumerate(range(0,self.day_range*settings.recurrence*24,self.day_range*24)):
#                 x[k][i]=sum(nested_list[j:j+self.day_range*24,i])  
#         return x
# =============================================================================
    #Increases demand rate over years
    def yearly_demand_rate(self, nested_list, settings):
        self.aux_list=nested_list
        for i in range(0,100000000):
            self.aux_list=np.r_[self.aux_list, nested_list*(1+(settings.ydr*(i+1)))]
            if len(self.aux_list) >= self.day_range*settings.recurrence*self.data_size:
                break
        return self.aux_list
    
    def replicate_data(self, nested_list, settings):
        self.aux_list=nested_list
        for i in range(0,100000000):
            self.aux_list=np.r_[self.aux_list, nested_list]
            if len(self.aux_list) >= self.day_range*settings.recurrence*self.data_size:
                break
        return self.aux_list
# network data ---------------------------------------------------------------------------------------------------------
class Network:
    def __init__(self, agent_data, gis_data): # agent_loc,
        """
        :param agent_data: AgentData object.
        :param agent_loc: dictionary mapping agent ids to node numbers
        :param gis_data: dataframe provided by GIS to us. has columns from_to (tuple), Losses total (W), length (m), total costs
        :output: a Network object with 2 properties: distance and losses (both n by n np.array). distance[1,3] is the
        distance from agent 1 to agent 3. has np.inf if cannot reach the agent.
        """
        
        # define distance and losses between any two agents in a matrix ----------------------------
        self.distance = np.inf * np.ones((agent_data.nr_of_agents, agent_data.nr_of_agents))
        for i in range(agent_data.nr_of_agents):
            self.distance[i, i] = 0.0   # distance to self is zero.
        for row_nr in range(len(gis_data["from_to"].values)):
            (From, To) = gis_data["from_to"].values[row_nr]
            self.distance[From, To] = gis_data.length.iloc[row_nr]

# =============================================================================
# Start here
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





























