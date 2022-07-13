--------------Input by user--------------
md -> str ('centralized' or 'decentralized')
horizon_basis -> str('weeks', 'months' or 'years')
recurrence -> int: If horizon_basis==weeks: recurrence<52; If horizon_basis==months: recurrence<12; If horizon_basis==years: recurrence<20
data_profile -> str('daily' or 'hourly')
yearly_demand_rate -> float (Ex: 0.05 == 5%)

--------------Input by Core functionalities/Knowledge Base-------------
agent_ids -> list of strings; a name for each agent
	Ex: agent_ids = ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"]
agent_types -> list of strings; each agent must have a type (producer, consumer or prosumer)
	Ex: agent_types = ["prosumer", "prosumer", "consumer", "producer"]

co2_emissions -> array with emissions by agent (float). 
	Ex:co2_emissions = np.array([1, 1.1, 0, 1.8])

The market module determines the period of simulation, based on the criteria of the “Horizon basis”, “Recurrence” and "Data Profile". 
For example, let us assume “Horizon basis” equals “week”, “Recurrence” equals 2 and "Data Profile" equals "hourly", it means that the 
total period of simulation is 336 hours (24 hours x 14 days). gmin, gmax, lmin, lmax, cost, util must match hours dimension.

gmin -> agents' minimum production over time; Dimensions: [hours x len(agent_ids)]
	Ex: gmin=np.zeros((336, len(agent_ids)))
gmax -> agents' maximum production over time; Dimensions: [hours x len(agent_ids)]
lmin -> agents' minimum consumption over time; Dimensions: [hours x len(agent_ids)]
lmax -> agents' maximum consumption over time; Dimensions: [hours x len(agent_ids)]
cost -> agent's production price over time; Dimensions: [hours x len(agent_ids)]
util -> agent's consumption price over time; Dimensions: [hours x len(agent_ids)]


---------------Input by GIS----------------
gis_data -> csv file from the GIS module; Example in the folder('Results_GIS')
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

class AgentData:
    """
    Object that stores all agent related data
    Contains an entry for each different input.
    If the input is constant in time, it is a dataframe with agent ID as column name, and the input as row
    If the input is varying in time, it is a dataframe with agent ID as column name, and time along the rows
    """
    def __init__(self, settings, name, a_type, gmin, gmax, lmin, lmax, cost, util, co2=None):
        # set nr of agents, names, and types
        self.nr_of_agents = len(name)
        # TODO print("todo make sure no ID is identical")
        self.agent_name = name
        self.agent_type = dict(zip(name, a_type))
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
             #checking data dimensions
            if len(gmin) + len(gmax) + len(lmin) + len(lmax) + len(cost) + len(util) != 6*self.day_range*settings.recurrence*self.data_size:
                raise ValueError('Data dimensions should be {:d}'.format(self.day_range*settings.recurrence*self.data_size))
            
            self.gmin = pd.DataFrame(gmin, columns=name)
            self.gmax = pd.DataFrame(gmax, columns=name)
            self.lmin = pd.DataFrame(lmin, columns=name)
            self.lmax = pd.DataFrame(lmax, columns=name)
    
            self.cost = pd.DataFrame(cost, columns=name)
            self.util = pd.DataFrame(util, columns=name)
            
        #Increasing data size if >8760 is required
        elif settings.horizon_basis == 'years':
            if settings.recurrence>20:
                raise ValueError('For horizon basis years, recurrence must not exceed 20')    
            if len(gmin) + len(gmax) + len(lmin) + len(lmax) + len(cost) + len(util) != 6*self.day_range*self.data_size:
                raise ValueError('Data dimensions should be {:d}'.format(self.day_range*self.data_size))
            
            lmin = self.yearly_demand_rate(lmin, settings) #demand increase
            lmax = self.yearly_demand_rate(lmax, settings) #demand increase
            gmin = self.replicate_data(gmin, settings)
            gmax = self.replicate_data(gmax, settings)
            
            cost = self.replicate_data(cost, settings)
            util = self.replicate_data(util, settings)
 
        # time dependent data -------------------------------------------------

       
            self.gmin = pd.DataFrame(gmin, columns=name)
            self.gmax = pd.DataFrame(gmax, columns=name)
            self.lmin = pd.DataFrame(lmin, columns=name)
            self.lmax = pd.DataFrame(lmax, columns=name)
    
            self.cost = pd.DataFrame(cost, columns=name)
            self.util = pd.DataFrame(util, columns=name)
     

#Increases demand rate over years
def yearly_demand_rate(self, nested_list, settings):
     self.aux_list=nested_list
     for i in range(0,100000000):
         self.aux_list=np.r_[self.aux_list, nested_list*(1+(settings.ydr*(i+1)))]
         if len(self.aux_list) >= self.day_range*settings.recurrence*self.data_size:
             break
     return self.aux_list


class Network:
    def __init__(self, agent_data, gis_data):
        """
        :param agent_data: AgentData object.
        :param gis_data: dataframe provided by GIS to us. has columns from/to (tuple), Losses total (W), length (m), total costs
        :output: a Network object with 2 properties: distance and losses (both n by n np.array). distance[1,3] is the
        distance from agent 1 to agent 3. has np.inf if cannot reach the agent.
        """

	def calculate_distances(graph, starting_vertex):
	#Calculates Dijkstra's shortest path between agents
	Input: graph: distances between agents. 'inf' if there's no possible link
		Ex: {0: {0: inf, 1: 1855.23241329466, 2: inf, 3: inf}, 1: {0: 1855.23241329466, 1: inf, 2: 1989.47106896218, 3: 1446.6888996543}, 2: {0: inf, 1: 1989.47106896218, 2: inf, 3: inf}, 3: {0: inf, 1: 1446.6888996543, 2: inf, 3: inf}}
		starting_vertex: starting vertex to calculate the distances to other agents
		Ex: 3
	Output: Distance from one agent to other. Dict where keys->agents and values->distance
		Ex (distance from agent 3 to other agents): {0: 3301.92131294896, 1: 1446.6888996543, 2: 3436.15996861648, 3: 0}


def make_centralized_market(name: str, agent_data: AgentData, settings: MarketSettings):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: str
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """

def make_decentralized_market(name: str, agent_data: AgentData, settings: MarketSettings, network: Network):
    """
    Makes the pool market, solves it, and returns a ResultData object with all needed outputs
    :param name: string, can give the resulting ResultData object a name.
    :param agent_data:
    :param settings:
    :return: ResultData object.
    """


class ResultData:
    def __init__(self, name, prob: cp.problems.problem.Problem,
                 cb: ConstraintBuilder,
                 agent_data: AgentData, settings: MarketSettings):
        """
        Object to store relevant outputs from a solved market problem.
        Initialization only extracts necessary values from the optimization
        Functions can be used to compute other quantities or generate plots
	Inputs:
        :param name: str. Same name as make_ _market
        :param prob: cvxpy problem object. Constraints and objective
        Ex: minimize Sum(param11660 @ Gn, None, False) + -Sum(param11661 @ Ln, None, False)
		subject to param11656 <= Gn
 	          Gn <= param11657
          	 param11658 <= Ln
           	Ln <= param11659
          	 Pn == Gn + -Ln
         	  -Sum(Pn, 1, False) == 0.0
        	   Gn[0, 0] == param11657[0, 0:4][0] @ b[0, 0]
         	  Sum(b([0, 1], 0), None, False) == 2.0 @ b[0, 0]
         	  Gn[1, 0] == param11657[1, 0:4][0] @ b[1, 0]
         	  Sum(b([0, 1], 0), None, False) == 2.0 @ b[0, 0]
          	 Gn[3, 3] == param11657[3, 0:4][3] @ b[3, 3]
          	 Sum(b([3, 4, 5, 6], 3), None, False) == 4.0 @ b[3, 3]
           	Gn[4, 3] == param11657[4, 0:4][3] @ b[4, 3]
         	  Sum(b([3, 4, 5, 6], 3), None, False) == 4.0 @ b[3, 3]
           	Gn[5, 3] == param11657[5, 0:4][3] @ b[5, 3]
           	Sum(b([3, 4, 5, 6], 3), None, False) == 4.0 @ b[3, 3]
           	Gn[6, 3] == param11657[6, 0:4][3] @ b[6, 3]
           	Sum(b([3, 4, 5, 6], 3), None, False) == 4.0 @ b[3, 3]
           	Gn[10, 3] == param11657[10, 0:4][3] @ b[10, 3]
           	Sum(b([10, 11], 3), None, False) == 2.0 @ b[10, 3]
           	Gn[11, 3] == param11657[11, 0:4][3] @ b[11, 3]
           	Sum(b([10, 11], 3), None, False) == 2.0 @ b[10, 3]
        :param cb: ConstraintBuilder used for cvxpy problem. Object from ConstraintBuilder
        :param agent_data: AgentData object
        :param settings: MarketSettings object
	Output:
	Dataframes Pn, Ln, Gn 
	Ex:     	prosumer_1  prosumer_2  consumer_1  producer_1
		0         -2.0         0.0        -1.0         3.0
		1         -2.0        -1.0         0.0         3.0
		2          0.0         1.0        -1.0         0.0
		3          0.0         0.0         0.0         0.0
		4          1.0         1.0        -3.0         1.0
		5          2.0         0.0        -3.0         1.0
		6         -2.0        -2.0        -1.0         5.0
		7         -2.0         0.0        -2.0         4.0
		8          1.0         0.0        -3.0         2.0
		9          0.0         0.0         0.0         0.0
		10         1.0         1.0        -3.0         1.0
		11         0.0        -1.0         0.0         1.0
	Dataframe shadow_price
	Ex:    		uniform price
		0              30
		1              24
		2              19
		3               0
		4              25
		5              31
		6              24
		7              32
		8              31
		9               0
		10             21
		11             33



        """

    def compute_output_quantities(self, settings, agent_data):
        #Qoe, social welfare ,settlement, ADG, SPM for different markets
	Output:
	Settlement
		Ex:    prosumer_1 prosumer_2 consumer_1 producer_1
		0      -240.0        0.0     -120.0      360.0
		1      -192.0      -96.0        0.0      288.0
		2         0.0       76.0      -76.0        0.0
		3         0.0        0.0        0.0        0.0
		4       100.0      100.0     -300.0      100.0
		5       248.0        0.0     -372.0      124.0
		6      -192.0     -192.0      -96.0      480.0
		7      -256.0        0.0     -256.0      512.0
		8       124.0        0.0     -372.0      248.0
		9         0.0        0.0        0.0        0.0
		10       84.0       84.0     -252.0       84.0
		11        0.0     -132.0        0.0      132.0	
	Social Welfare
		Ex:
  			 Social Welfare
			0           -59.0
			1           -68.0
			2           -97.0
			3             0.0
			4          -102.0
			5           -72.0
			6          -145.0
			7          -111.0
			8          -178.0
			9             0.0
			10          -75.0
			11         -106.0
	QoE
		Ex:
			            QoE
			0      0.809861
			1      0.804892
			2      0.809776
			3   Not Defined
			4      0.749632
			5      0.829688
			6      0.815514
			7      0.808895
			8      0.822485
			9   Not Defined
			10     0.818883
			11     0.770047
	
	ADG and SPM
		Ex:     prosumer_1 prosumer_2 producer_1
		ADG  50.372329  48.015424  30.374496
    			prosumer_1 prosumer_2 producer_1
		SPM  63.333333  53.333333       45.0


def find_best_price(self, period: int, agent_name, agent_data, settings):
#Defines the best bidding price for agent 'agent_name' in time 'period'; Period and agent_name must be selected by the user














Conventions:
1. In matrix notation, rows iterate over time, columns iterate over agents. So Pmin[1,5] is time 1 agent 5.
2. Index for an agent is denoted by "i", index for time is "t", index for node is "n", for pipe is "p"
3. Power injection by an agent or node is POSITIVE when it is a net generator.
4. Trade is positive when energy is sold
5. If a 3-dim array would be needed, cvxpy cannot handle this. The solution we use is a list of matrix variables.
    For example, trades are needed for each hour, and agent pair. The list Tnm contains a matrix variable for each t
    So Tnm[t][i,j] is the trade from i to j at time t.
