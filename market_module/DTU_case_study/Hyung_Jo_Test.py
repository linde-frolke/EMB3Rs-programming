###
'''
Test case: Pool 
md: market design
nr_of_hours: 48hrs
start_datetime: dd-mm-yyyy
offer_type: 'Simple','Block',and 'energyBudget'
prod_diff: noPref, co2Emissions, networkDistance, losses
el_dependent: True/False (only for Pool)
el_price: if el_dependent=True, price for all market hours
Network: 'none','direction'
Objective: if md=community, 'autonomy','peakShaving'
Community settings: 
agent_ids[n]: id of each agent imported from TEO
co2_emission[n]: CO2 emssions by agent, list (only when co2Emissions are selected in P2P)
gmax[t,n]: max values for time and each generator, list
lmax[t,n]: max values for time and each consumer, list
cost[t,n]: minimum price generators want to receive
util[t]: bid price of consumption
gis_data: network data, dictionary (linked agents,total length between them,total costs assoicated with each pipeline)
# check gis module
block_offer: dictionary with agent IDs as keys, agents that submit block bids
is_in_community: boolean if agent is in community or not
is_chp: boolean specifies if CHP or not
chp_pars: parameters for CHP
'''
###
"""
:param input_dict: could be like this:

input_dict = {'md': 'pool',  # other options are  'p2p' or 'community'
                'nr_of_hours': 12,
                'offer_type': 'simple',
                'prod_diff': 'noPref',
                'network': 'none',
                'el_dependent': 'false',
                'el_price': 'none',
                'agent_ids': ["prosumer_1",
                        "prosumer_2", "consumer_1", "producer_1"],
                'objective': 'none', # objective for community
                'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, # or values instead
                'gmax': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                'lmax': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                'cost': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                'util': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                'is_in_community': 'none',  # allowed values are 'none' or boolean array of size (nr_of_agents) 
                'block_offer': 'none',  # allowed values are 'none' or list of lists for each agent,
                            e.g {'prosumer_1': [[0, 1]], 'producer_1': [[3, 4, 5, 6], [10, 11]]}
                'is_chp': 'none', # allowed values are 'none' or a list with ids of agents that are CHPs
                'chp_pars': 'none',  # a dictionary of dictionaries, including parameters for each agent in is_chp.
                                                    # {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                    #  'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
                #defaults = pd.DataFrame({"col": {"alpha": default_alpha, "r": 0.45, "rho_H": 0.9, "rho_E": 0.25}})
                #alpha = fuel price, r: min. pwr-t-heat, rho_H: Fuel eff. for el. gen, rho_E: Fuel eff. for heat gen
                'gis_data': 'none'  # or dictionary of format: 
                        # {'From/to': [(0, 1), (1, 2), (1, 3)],
                        #  'Losses total [W]': [22969.228855, 24122.603833, 18138.588662],
                        #  'Length': [1855.232413, 1989.471069, 1446.688900],
                        #  'Total_costs': [1.848387e+06, 1.934302e+06, 1.488082e+06]},
                'nodes' : "none", 
                "edges" : "none"
                }
"""
from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market

import numpy as np
import pandas as pd
import networkx as nx
#from pyvis.network import Network
from datetime import datetime, timedelta

# define function to load variables
def load_data(nr_h,Year,Month,Day):
    path = "C:/Users/hyung/Documents/Desktop/Student Job"
    consumption_data_pre = pd.read_csv(path+'/Data/EMB3Rs_Full_Data.csv',
                                        usecols=[i for i in range(1,23)],
                                        parse_dates=['Hour'],
                                        index_col = ['Hour']
                                        )

    start_date = consumption_data_pre.loc[f'{Year}-{Month}-{Day}'].index[0]
    end_date = start_date + timedelta(hours=nr_h)                                 
    consumption_data = consumption_data_pre[start_date:end_date]
    network_data = pd.read_csv(path+'/Data/Network_data_&_preferences._/Nodes.csv',dtype={'Id':np.int64})
    nodes_name_data = pd.read_csv(path + '/Data/Network_data_&_preferences._/Nodes_data.csv',names=['id','name'])
    buildingID = pd.read_csv(path+'/Data/Network_data_&_preferences._/BuildingID.csv',names=['Node'])
    pipe_dir = pd.read_csv(path+'/Data/Network_data_&_preferences._/Pipe_edges.csv',names=['from','to'])
    return consumption_data, network_data, nodes_name_data, buildingID, pipe_dir

# costs of data
def cost_chp(n_CHPs,n_hours,p_length,Year,Month,Day):
    # indices
    # Year: Year that wants to be observed
    # Month: Starting month
    # Day: Starting day
    #n_CHPs = 13       #representing each of the 13 CHPs
    #n_hours = 24*365  #no. of hours in simulation (also time steps for CHPs)
    
    # inputs
    # Electricity price from NordPool, using UTC time and DKK as price, in DKK/MWh
    df_elprice = pd.read_csv('C:/Users/hyung/Documents/Desktop/Student Job/Ida - Codes (CHP pricing generator)/excess-heat-in-market-main/Data/elspot_2018-2020_DK2.csv',
                        index_col = 2, parse_dates=True)
    
    start_date = df_elprice.loc[f'{Year}-{Month}-{Day}'].index[-1]
    end_date = start_date + timedelta(hours=n_hours)
    
    # periods for average temperatures
    #p_length = 6
    #n_periods = int((n_hours/p_length))
    
    #periods = [(int((p-1)*p_length + 1), int(p*p_length))
    #           for p in range(1,n_periods+1)]
    
    # create a list of timestamps for plotting or other interesting things
    time_list_hourly = []
    time_list_hourly = [start_date + timedelta(hours=t) for t in range(1,n_hours+1)]
    
    price_el_hourly = df_elprice.loc[end_date-timedelta(hours=1):start_date,'SpotPriceDKK'].iloc[::-1].values # time is reversed from most recent to oldest 
    ## PARAMETERS

    # fuel price for fuel used for each hour for each plant [DKK/MWh]
    # taken from Ommen, Markussen & Elmegaard 2013: Heat pumps in district heating networks [€/GJ]
    # the €-price is multiplied by 7.5/0.278 to get the DKK-price and the MWh-quantity
    price_fuel = np.array([6.5, 2, 7.3, 7.3, 7.3, 2, 3.5, 6.9, 2, 2, 2, 6.5, 6.5])*7.5/277.8
    
    # power-to-heat ratio for each generator, assumed to be 0.45 for all
    # (COMBINED HEAT AND POWER (CHP) GENERATION Directive 2012/27/EU of the European
    # Parliament and of the Council Commission Decision 2008/952/EC)
    phr = np.array([0.45]*n_CHPs)
    
    # fuel efficiency for producing heat and electricity per plant [t fuel/MWh el or heat]
    # from Ommen, Markussen & Elmegaard 2013: Heat pumps in district heating networks
    # assuming ρ_el = 0.2 and ρ_heat = 0.9 for the ones without information
    #eff_el = eff_heat = repeat([1], outer=length(CHPs))
    eff_heat = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.83, 0.91, 0.93, 0.81, 0.99, 0.99, 0.9, 0.9])
    eff_el = np.array([0.21, 0.2, 0.18, 0.29, 0.2, 0.19, 0.36, 0.43, 0.18, 0.12, 0.18, 0.2, 0.2])
    
    ## CHP bids
    #cost_heat = [price_el_hourly[t] <= price_fuel[i]*eff_el[i] ?
    #    price_fuel[i]*(eff_el[i]*phr[i]+eff_heat[i]) - price_el_hourly[t]*phr[i] :
    #    price_el_hourly[t]*eff_heat[i]/eff_el[i] for i in 1:n_CHPs, t in 1:n_hours]
    
    cost_heat = np.array([price_fuel[i]*(eff_el[i]*phr[i]+eff_heat[i]) - price_el_hourly[t]*phr[i] 
                 if price_el_hourly[t] <= price_fuel[i]*eff_el[i] else price_el_hourly[t]*eff_heat[i]/eff_el[i]
                 for i in range(0,n_CHPs) for t in range(0,n_hours) 
                 ]).reshape(n_CHPs,n_hours)
    cost_heat = pd.DataFrame(cost_heat)
    
    #cost_heat.T.plot.line()
    #plt.show()
    return cost_heat, price_el_hourly

# settings
start_day = 40
nr_h = 48
grid_max_import_kWh_h = 10**5
nr_con = 21
nr_CHPs = 13
p_length = 1
Year = 2018
Month = 5
Day = 15

# Load data
consumption_data, network_data, nodes_names_data, buildingID, pipe_dir = load_data(nr_h,Year,Month,Day)

# agent ids
consumer_ids = [f'consumer_{i}' for i in range(1,len(consumption_data.columns)+1)]
generator_ids = [f'generator_{i}' for i in range(1,nr_CHPs+1)]
agent_ids = np.concatenate([consumer_ids,generator_ids]).tolist()

# maximum capacity of generator
# [[nr_generator],[]...[]*nr_hours]
# (nr_hours,nr_agents)
g_cap = np.zeros(len(agent_ids))
g_cap[nr_con:] = [251, 400, 240, 250, 94, 190, 331, 585, 96.8, 69, 73, 41.8, 53]
g_max = np.tile(g_cap,(nr_h,1)).tolist()

# Only take 21 consumption from EMB3rs full data 
l_cap = np.zeros(len(agent_ids))
l_cap[:nr_con] = consumption_data.max(axis=0).tolist()
l_max = np.tile(l_cap,(nr_h,1)).tolist()

# Cost of generators
# costs = nr_h x nr_agent
cost = np.zeros((nr_h,len(agent_ids)))
costs_chp,price_el_hourly = cost_chp(nr_CHPs,nr_h,p_length,Year,Month,Day)

cost[:,nr_con:] = costs_chp.T.values.tolist()
cost = cost.tolist()
# Utility costs

util_cost = np.zeros(len(agent_ids))
max_cost = max(cost.max(axis=1)) # max cost of generator cost, so set to sufficiently high utility
util_cost[:nr_con] = max_cost
# maximum price of generator from cost_chp
utility = np.tile(util_cost,(nr_h,1)).tolist()

'''
'chp_pars': 'none',  # a dictionary of dictionaries, including parameters for each agent in is_chp.
                                                    # {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                    #  'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
'''

chp_pars = {}
# not really needed since I used Ida's code
# defaults = pd.DataFrame({"col": {"alpha": default_alpha, "r": 0.45, "rho_H": 0.9, "rho_E": 0.25}})
price_fuel = list(np.array([6.5, 2, 7.3, 7.3, 7.3, 2, 3.5, 6.9, 2, 2, 2, 6.5, 6.5])*7.5/0.278)
phr = np.repeat([0.45], nr_CHPs)
eff_heat = [0.9, 0.9, 0.9, 0.9, 0.9, 0.83, 0.91, 0.93, 0.81, 0.99, 0.99, 0.9, 0.9]
eff_el = [0.21, 0.2, 0.18, 0.29, 0.2, 0.19, 0.36, 0.43, 0.18, 0.12, 0.18, 0.2, 0.2]

for i,chp in enumerate(generator_ids):
        chp_pars[chp] = {"alpha": price_fuel[i], 'r' :phr[i], 'rho_H': eff_heat[i], 'rho_E': eff_el[i]}


#create network data
paper = []
nodes_name_dict = dict(zip((nodes_name_data.id),nodes_name_data.name))
for node_ in nodes_name_dict.keys():
    paper.append(f'{nodes_name_dict[node_]}')
    
buildingNames = [f'{id[0]}' for id in buildingID.values]

g_paper = nx.DiGraph()
for i,node in enumerate(paper):
    #print(names_dict[node])
    if node in buildingNames:
        g_paper.add_node(node,color='red')
    elif node == '42':
        g_paper.add_node(node,color='green')
    else:
        g_paper.add_node(node,color='blue')
        
for index,row_edge in pipe_dir.iterrows():
    edge1 = '{}'.format(row_edge['from'])
    edge2 = '{}'.format(row_edge['to'])
    g_paper.add_edge(edge1,edge2)


# nodes and edges
node = list(g_paper.nodes)
edge = list(g_paper.edges)

pipe_length = np.ones(len(g_paper.edges))
# pipe_losses = how to calculate the total loss?
# pipe_cost = how to calculate the total cost?


input_dict = {
        'md': 'pool',  # other options are  'p2p' or 'community'
        'nr_of_hours': nr_h,
        'offer_type': 'simple',
        'prod_diff': 'noPref',
        'network': 'none',
        'el_dependent': 'true',
        'el_price': price_el_hourly, # not list but array 
        'agent_ids': agent_ids,
        'objective': 'none', 
        'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
        'gmax': g_max, 
        'lmax': l_max, 
        'cost': cost, 
        'util': utility, 
        'co2_emissions': 'none',
        'is_in_community': 'none', 
        'block_offer': 'none', 
        'is_chp': generator_ids, 
        'chp_pars': chp_pars,  
        'gis_data': 'none',  
        'nodes': 'none', 
        'edges': 'none',
}

results = run_shortterm_market(input_dict=input_dict) 

