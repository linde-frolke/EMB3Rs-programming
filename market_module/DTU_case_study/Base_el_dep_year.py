from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market

import numpy as np
import pandas as pd
from datetime import timedelta

from market_module.DTU_case_study.load_data import load_data
from market_module.DTU_case_study.save_tocsv import save_tocsv


# Pool Base
##---------------------------

# settings
tot_h = 24*365
nr_h = 24
grid_max_import_kWh_h = 10**5
nr_con = 30
nr_grid = 1
nr_sm = 1
nr_agents = nr_con+nr_grid+nr_sm
cap_constant = 1.5
util_constant = 1.1
Year = 2018
Month = 4
Day = 14
COP = 3.5

# Load data
consumption_data, price_el_hourly, grid_price = load_data(tot_h,Year,Month,Day)

# agent ids
grid_ids = ['grid_1']
sm_ids = ['sm_1']
consumer_ids = [f'consumer_{i}' for i in range(1,len(consumption_data.loc[:,'Row_House1':'Row_House30'].columns)+1)]
agent_ids = np.concatenate([grid_ids,sm_ids,consumer_ids]).tolist()

# maximum capacity of generators, gmax
# (nr_hours,nr_agents)
# take the largest consumption value
max_consum = np.ones(tot_h)*max(consumption_data.max(axis=0)[:-1])*cap_constant
max_sm_avaliable = (consumption_data.loc[:,'SM_avail_heat'])

g_max = np.zeros((tot_h,nr_agents))
g_max[:,agent_ids.index('sm_1')] = max_sm_avaliable
g_max[:,agent_ids.index('grid_1')] = max_consum


# maximum capacity of demands, lmax
l_max = np.zeros((tot_h,nr_agents))

sm_consumption = consumption_data.loc[:,'SM_consumption']
l_max[:,agent_ids.index('sm_1')] = sm_consumption
l_max[:,nr_grid+nr_sm:] = consumption_data.loc[:,'Row_House1':'Row_House30']

# Cost of generators
# costs = nr_h x nr_agent
# cost of grid = grid_price
# cost of supermarket = el_price/COP
# cost of demands = 0
cost_h = price_el_hourly/COP
cost = np.zeros((tot_h,nr_agents))
cost[:,agent_ids.index('sm_1')] = cost_h.values.squeeze()
cost[:,agent_ids.index('grid_1')] = grid_price.values.squeeze() # input prices of the grid generatred from Ida's code

# Utility costs
util_cost = np.zeros(nr_agents)
max_grid = grid_price.max()[0]# max cost of generator cost, so set to sufficiently high utility
util_cost[nr_grid:] = max_grid*util_constant
# maximum price of generator from cost_chp
utility = np.tile(util_cost,(tot_h,1))


#for loop, optimizies it for every 24 hours
# g_max,l_max,l_min,cost,utility has to be for every 24
# index it so it skips 24 index after first iteration in loop
start_date = consumption_data.index.date[0]
end_date = consumption_data.index.date[-1]+timedelta(days=1)
one_year_idx = pd.date_range(start=start_date,end=end_date,freq='D')[:-1]
time_range = pd.date_range(start=start_date,end=end_date,freq='H')[:-1]

# monthly average
uniform_price_year = []
Pn_year = []
Gn_year = []
Ln_year = []

sw_year = []
settlement_year = []
Gn_revenue_year = []
Ln_revenue_year = []


for i,date in enumerate(one_year_idx,1):
	print('-'*80)
	print(date)
	print('-'*80)
	
	g_max_list = g_max[24*(i-1):24*i,:].tolist()
	l_max_list = l_max[24*(i-1):24*i,:].tolist()

	cost_list = cost[24*(i-1):24*i,:].tolist()
	utility_list = utility[24*(i-1):24*i,:].tolist()

	input_dict = {
		'md': 'pool',  # other options are  'p2p' or 'community'
		'nr_of_hours': nr_h,
		'offer_type': 'simple',
		'prod_diff': 'noPref',
		'network': 'none',
		'el_dependent': 'false',
		'el_price': 'none', # not list but array 
		'agent_ids': agent_ids,
		'objective': 'none', 
		'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
		'gmax': g_max_list, 
		'lmax': l_max_list,
		'cost': cost_list, 
		'util': utility_list, 
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

	uniform_price = pd.DataFrame(results['shadow_price']['uniform price'])

	Pn = pd.DataFrame.from_dict(results['Pn'])
	Gn = pd.DataFrame.from_dict(results['Gn'])
	Ln = pd.DataFrame.from_dict(results['Ln'])

	sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
	settlement = pd.DataFrame.from_dict(results['settlement'])

	Gn_revenue = pd.DataFrame(uniform_price.values*Gn.values, columns=Gn.columns, index=Gn.index)
	Ln_revenue = pd.DataFrame(uniform_price.values*Ln.values, columns=Ln.columns, index=Ln.index)

	uniform_price_year.append(uniform_price)
	Pn_year.append(Pn)
	Gn_year.append(Gn)
	Ln_year.append(Ln)
	sw_year.append(sw)
	settlement_year.append(settlement)
	Gn_revenue_year.append(Gn_revenue)
	Ln_revenue_year.append(Ln_revenue)

# Save and export dataframes to CSV
model_name = 'Pool_base'
save_tocsv(model_name,time_range,uniform_price_year,Pn_year,Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year)