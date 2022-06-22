from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
#from pyvis.network import Network
import unicodedata
from market_module.DTU_case_study.plotting import plotting
from market_module.DTU_case_study.load_data import load_data

# settings
model_name = 'base_el_dep'
nr_h = 48
grid_max_import_kWh_h = 10**5
nr_con = 30
nr_grid = 1
nr_sm = 1
nr_agents = nr_con+nr_grid+nr_sm
cap_constant = 1.5
Year = 2019
Month = 3
Day = 10
COP = 3.5

# Load data
consumption_data, price_el_hourly, grid_price = load_data(nr_h,Year,Month,Day)

# agent ids
grid_ids = ['grid_1']
sm_ids = ['sm_1']
consumer_ids = [f'consumer_{i}' for i in range(1,len(consumption_data.loc[:,'Row_House1':'Row_House30'].columns)+1)]
agent_ids = np.concatenate([grid_ids,sm_ids,consumer_ids]).tolist()

#
# maximum capacity of generators, gmax
# (nr_hours,nr_agents)
# take the largest consumption value
max_consum = np.ones(nr_h)*max(consumption_data.max(axis=0)[:-1])*cap_constant
max_sm_avaliable = (consumption_data.loc[:,'SM_avail_heat'])

g_max = np.zeros((nr_h,nr_agents))
g_max[:,agent_ids.index('sm_1')] = max_sm_avaliable
g_max[:,agent_ids.index('grid_1')] = max_consum
g_max = g_max.tolist()

# maximum capacity of demands, lmax
l_max = np.zeros((nr_h,nr_agents))
l_min = np.zeros((nr_h,nr_agents))

sm_consumption = consumption_data.loc[:,'SM_consumption']
l_max[:,agent_ids.index('sm_1')] = sm_consumption
l_min[:,agent_ids.index('sm_1')] = sm_consumption
l_max[:,nr_grid+nr_sm:] = consumption_data.loc[:,'Row_House1':'Row_House30']
l_min[:,nr_grid+nr_sm:] = consumption_data.loc[:,'Row_House1':'Row_House30']
l_max = l_max.tolist()
l_min = l_min.tolist()
# Cost of generators
# costs = nr_h x nr_agent
# cost of grid = grid_price
# cost of supermarket = el_price/COP
# cost of demands = 0
cost_h = price_el_hourly/COP
cost = np.zeros((nr_h,nr_agents))
cost[:,agent_ids.index('sm_1')] = cost_h
cost[:,agent_ids.index('grid_1')] = grid_price.values.squeeze() # input prices of the grid generatred from Ida's code
cost = cost.tolist()

# Utility costs
util_cost = np.zeros(nr_agents)
max_grid = grid_price.max()[0]# max cost of generator cost, so set to sufficiently high utility
util_cost[nr_grid:] = max_grid
# maximum price of generator from cost_chp
utility = np.tile(util_cost,(nr_h,1)).tolist()

input_dict = {
        'md': 'pool',  # other options are  'p2p' or 'community'
        'nr_of_hours': nr_h,
        'offer_type': 'energyBudget',
        'prod_diff': 'noPref',
        'network': 'none',
        'el_dependent': 'false',
        'el_price': 'none', # not list but array 
        'agent_ids': agent_ids,
        'objective': 'none', 
        'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
        'gmax': g_max, 
        'lmax': l_max,
        'lmin': l_min, 
        'cost': cost, 
        'util': utility, 
        'co2_emissions': 'none',
        'is_in_community': 'none', 
        'block_offer': 'none', 
        'is_chp': 'none', 
        'chp_pars': 'none',  
        'gis_data': 'none',  
        'nodes': 'none', 
        'edges': 'none',
}

results = run_shortterm_market(input_dict=input_dict) 

plotting(results, model_name)


