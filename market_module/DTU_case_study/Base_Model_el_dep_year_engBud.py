from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

#from pyvis.network import Network
import unicodedata

from market_module.DTU_case_study.plotting import plotting
from market_module.DTU_case_study.load_data import load_data

# settings
model_name = 'base_el_dep'
tot_h = 24*365
nr_h = 24
grid_max_import_kWh_h = 10**5
nr_con = 30
nr_grid = 1
nr_sm = 1
nr_agents = nr_con+nr_grid+nr_sm
cap_constant = 1.5
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
l_max[:,agent_ids.index('sm_1')] = sm_consumption*2 # for upward and downward flexibility

l_max[:,nr_grid+nr_sm:] = consumption_data.loc[:,'Row_House1':'Row_House30']*2

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
util_cost[nr_grid:] = max_grid
# maximum price of generator from cost_chp
utility = np.tile(util_cost,(tot_h,1))


#for loop, optimizies it for every 24 hours
# g_max,l_max,l_min,cost,utility has to be for every 24
# index it so it skips 24 index after first iteration in loop
start_date = consumption_data.index.date[0]
end_date = consumption_data.index.date[-1]+timedelta(days=1)
one_year_idx = pd.date_range(start=start_date,end=end_date,freq='D')
time_range = pd.date_range(start=start_date,end=end_date,freq='H')[:-1]

# monthly average
uniform_price_year = []
Pn_year = []
Gn_year = []
Ln_year = []

sw_year = []
settlement_year = []


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
		'offer_type': 'energyBudget',
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

	uniform_price_year.append(uniform_price)
	Pn_year.append(Pn)
	Gn_year.append(Gn)
	Ln_year.append(Ln)
	sw_year.append(sw)
	settlement_year.append(settlement)


df_uniform_price = pd.concat(uniform_price_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()
df_Pn_year = pd.concat(Pn_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()
df_Gn_year = pd.concat(Gn_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()
df_Ln_year = pd.concat(Ln_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()
df_sw_year = pd.concat(sw_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()
df_settlement_year = pd.concat(settlement_year).set_index(time_range).groupby(pd.Grouper(freq='24H')).mean()

# Monthly Averages
df_uniform_price_mavg = pd.concat(uniform_price_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()
df_Pn_year_mavg = pd.concat(Pn_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()
df_Gn_year_mavg = pd.concat(Gn_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()
df_Ln_year_mavg = pd.concat(Ln_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()
df_sw_year_mavg = pd.concat(sw_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()
df_settlement_year_mavg = pd.concat(settlement_year).set_index(time_range).groupby(pd.Grouper(freq='1M')).mean()


#plotting(results, model_name)
Pn = df_Pn_year
Gn = df_Gn_year
Ln = df_Ln_year

sw = df_sw_year*-1 # social welfare should be flipped
settlement = df_settlement_year

model_name = 'Energy Budget-Electricity Dependent'
# Plotting 
# Production and consumption of supermarket 
fig, axes = plt.subplots()
Ln.sm_1.plot(ax=axes)
Gn.sm_1.plot(ax=axes)
axes.set_title(f'Supermarket Consumption and Production ({model_name})')
axes.legend(['Consumption','Production'])
#axes.legend(['Consumption,Production'])
axes.set_xlabel('Time')
axes.set_ylabel('kWh')
axes.grid()

# Production of grid and Market Price
fig, axes = plt.subplots(2,1,sharex=True)
axes[0].plot(Gn['grid_1'])
axes[1].plot(df_uniform_price)
axes[0].set_title(f'Grid Production ({model_name})')
axes[1].set_title(f'Market Price ({model_name})')
axes[1].set_xlabel('Time')
axes[0].grid()
axes[1].grid()

# Production of Grid and SuperMarket
fig,axes = plt.subplots()
Pn.grid_1.plot(ax=axes)
Pn.sm_1.plot(ax=axes)
axes.legend(['Grid','Supermarket'])
axes.set_title(f'$P_n$ of Agents ({model_name})')
axes.set_ylabel('kWh')
axes.grid()

# Social Welfare
start = sw.min()[0]
end = sw.max()[0]
fig,axes = plt.subplots()
sw.plot(ax=axes)
axes.yaxis.set_ticks(np.arange(start,end,-(start-end)/10.0))
axes.legend(['Social Welfare'],loc='upper left')
axes.set_xlabel('Time')
axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
axes.set_title(f'Social Welfare ({model_name})')
axes.grid()

# Settlement
fig,axes = plt.subplots()
settlement.grid_1.plot(ax=axes)
settlement.sm_1.plot(ax=axes)
axes.set_xlabel('Time')
axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
axes.set_title(f'Settlement ({model_name})')
axes.grid()
axes.legend()

# compare actual consumption of supermarket and flexiblity
Ln_sm = pd.concat(Ln_year).set_index(time_range)['sm_1']
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(sm_consumption,alpha=0.5,label='Actual',linestyle='-', marker='o',markersize=2)
ax.plot(Ln_sm,alpha=0.5,label='Flexible',linestyle='-', marker='o',markersize=2)
#ax.scatter(x=sm_consumption.index,y=sm_consumption.values,s=10,alpha=0.3,label='Actual')
#ax.scatter(x=Ln_sm.index,y=Ln_sm.values,s=10,alpha=0.3,label='Flexible')
ax.set_xlabel('Time')
ax.set_ylabel('kWh')
ax.set_title('Load flexibility')
ax.legend()
plt.tight_layout()
plt.grid(which='minor')
plt.grid(which='major')
plt.show()


'''
# Monthly Averages
fig, ax = plt.subplots()
df_uniform_price_mavg.plot.bar(ax=ax,label='Uniform Price')
ax.xticks(rot=30)
ax.set_xticklabels(df_uniform_price_mavg.index.month_name(),rotation=30)
ax.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax.set_title('Monthly Average Uniform Pricing')
plt.legend()

fig, ax = plt.subplots()
df_Pn_year_mavg.plot.bar() 
ax.xticks(rot=30)
ax.set_xticklabels(df_uniform_price_mavg.index.month_name(),rotation=30)
ax.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax.set_title('Monthly Average Uniform Pricing')

fig, ax = plt.subplots()
ax.boxplot(df_Gn_year.sm_1)
ax.set_xticklabels(df_Gn_year_mavg.index.month_name(),rotation=30)
ax.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax.set_title('Monthly Average Production of SuperMarket')

plt.legend()
plt.show()

fig, ax = plt.subplots()
df_Ln_year_mavg.plot.bar() 
ax.xticks(rot=30)
ax.set_xticklabels(df_uniform_price_mavg.index.month_name(),rotation=30)
ax.set_ylabel('kWh')
ax.set_title('Monthly Average Uniform Pricing')

fig, ax = plt.subplots()
df_sw_year_mavg.plot.bar()  
ax.xticks(rot=30)
ax.set_xticklabels(df_uniform_price_mavg.index.month_name(),rotation=30)
ax.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax.set_title('Monthly Average Uniform Pricing')

fig, ax = plt.subplots()
df_settlement_year_mavg.plot.bar() 
ax.xticks(rot=30)
ax.set_xticklabels(df_uniform_price_mavg.index.month_name(),rotation=30)
ax.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax.set_title('Monthly Average Uniform Pricing')
plt.show()
'''