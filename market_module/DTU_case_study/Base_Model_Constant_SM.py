from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import unicodedata
#from pyvis.network import Network
from datetime import datetime, timedelta

# define function to load variables
def load_data(nr_h,Year,Month,Day):
    path = "C:/Users/hyung/Documents/Desktop/Student Job"
    consumption_data_pre = pd.read_csv(path+'/Data/EMB3Rs_Full_Data.csv',
                                        usecols=range(1,34),
                                        parse_dates=['Hour'],
                                        index_col = ['Hour']
                                        )

    start_date = consumption_data_pre.loc[f'{Year}-{Month}-{Day}'].index[0]
    end_date = start_date + timedelta(hours=nr_h-1)                                 
    consumption_data = consumption_data_pre[start_date:end_date]

    df_elprice = pd.read_csv(path + '/Ida - Codes (CHP pricing generator)/excess-heat-in-market-main/Data/elspot_2018-2020_DK2.csv',
                        index_col = 2, parse_dates=True)
    df_elprice[::-1].loc[start_date:end_date,'SpotPriceDKK'].values # time is reversed from most recent to oldest 

    grid_price = pd.read_csv(path + '/Data/df_grid_price.csv',usecols=[1])[:nr_h]
    grid_price = grid_price.set_index(consumption_data.index)
    return consumption_data, price_el_hourly, grid_price


def plotting(results):
        #time_range = pd.date_range(start=f'{Year}-{Month}-{Day}',periods=nr_h,freq='H')
        uniform_price = results['shadow_price']['uniform price']
        Pn = pd.DataFrame.from_dict(results['Pn'])
        Gn = pd.DataFrame.from_dict(results['Gn'])
        Ln = pd.DataFrame.from_dict(results['Ln'])
        sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
        settlement = pd.DataFrame.from_dict(results['settlement'])
        shadow_price = pd.DataFrame.from_dict(results['shadow_price'])

        # Production and consumption of supermarket 
        fig, axes = plt.subplots()
        Ln.sm_1.plot(ax=axes)
        Gn.sm_1.plot(ax=axes)
        axes.set_title('Super Market Consumption and Production')
        axes.legend(['Consumption','Production'])
        #axes.legend(['Consumption,Production'])
        axes.set_xlabel('Time[h]')
        axes.set_ylabel('kWh')

        # Production of grid and Market Price
        fig, axes = plt.subplots(2,1,sharex=True)
        axes[0].plot(Gn['grid_1'])
        axes[1].plot(shadow_price['uniform price'])
        axes[0].set_title('Grid Production')
        axes[1].set_title('Market Price')
        axes[1].set_xlabel('Time [h]')

        # Production of Grid and SuperMarket
        fig,axes = plt.subplots()
        Pn.grid_1.plot(ax=axes)
        Pn.sm_1.plot(ax=axes)
        axes.legend(['Grid','Super Market'])
        axes.set_title('$P_n$ of Agents')

        # Social Welfare
        fig,axes = plt.subplots()
        sw.plot(ax=axes)
        axes.legend(['Social Welfare'],loc='upper left')
        axes.set_xlabel('Time [h]')
        axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
        axes.set_title('Social Welfare')

        # Settlement
        fig,axes = plt.subplots()
        settlement.grid_1.plot(ax=axes)
        settlement.sm_1.plot(ax=axes)
        axes.set_xlabel('Time [h]')
        axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
        axes.legend()
        plt.show()



# settings
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

# Load data
consumption_data, price_el_hourly, grid_price = load_data(nr_h,Year,Month,Day)

# agent ids
grid_ids = ['grid_1']
sm_ids = ['sm_1']
consumer_ids = [f'consumer_{i}' for i in range(1,len(consumption_data.loc[:,'Row_House1':'Row_House30'].columns)+1)]
agent_ids = np.concatenate([grid_ids,sm_ids,consumer_ids]).tolist()

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

sm_consumption = consumption_data.loc[:,'SM_consumption']
l_max[:,agent_ids.index('sm_1')] = sm_consumption
l_max[:,nr_grid+nr_sm:] = consumption_data.loc[:,'Row_House1':'Row_House30']
l_max = l_max.tolist()

# Cost of generators
# costs = nr_h x nr_agent
# cost of grid = grid_price
# cost of supermarket = 0
# cost of demands = 0
cost = np.zeros((nr_h,nr_agents))
cost[:,:nr_grid] = grid_price.values # input prices of the grid generatred from Ida's code
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
        'offer_type': 'simple',
        'prod_diff': 'noPref',
        'network': 'none',
        'el_dependent': 'false',
        'el_price': 'none', # not list but array 
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
        'is_chp': 'none', 
        'chp_pars': 'none',  
        'gis_data': 'none',  
        'nodes': 'none', 
        'edges': 'none',
}

results = run_shortterm_market(input_dict=input_dict) 

plotting(results)
