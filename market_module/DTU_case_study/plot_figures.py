#%%
import sys 
sys.path.append('C://Users//hyung//Documents//GitHub//EMB3Rs-programming')
#%%
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import numpy as np
from market_module.DTU_case_study.load_data import load_data

plt.rcParams["figure.figsize"] = (14,6)

#Path to load the dataframe from
path = 'C://Users//hyung//Documents//GitHub//EMB3Rs-programming//market_module//DTU_case_study//Data//'

modelName_base = 'PoolNetwork_base'
modelName_EB = 'PoolNetwork_EB'

# Load flexibility
Year = 2018
Month = 4
Day = 14
tot_h = 365*24
COP = 2.5 
nr_h = 24

consumption_data, price_el_hourly, grid_price = load_data(tot_h,Year,Month,Day)

# agent ids
grid_ids = ['grid_1']
sm_ids = ['sm_1']
consumer_ids = [f'consumer_{i}' for i in range(1,len(consumption_data.loc[:,'Row_House1':'Row_House30'].columns)+1)]
agent_ids = np.concatenate([grid_ids,sm_ids,consumer_ids]).tolist()

# Time index
time_range_h = pd.date_range(start=consumption_data.index[0], end=consumption_data.index[-1],freq='H')
time_range_d = pd.date_range(start=consumption_data.index[0], end=consumption_data.index[-1],freq='D')
time_range_m = pd.date_range(start=consumption_data.index[0], end=consumption_data.index[-1]+timedelta(days=30),freq='M')

# Pool model
output_names = ['uniform_price','Pn','Gn','Ln','sw','settlement','Gn_revenue','Ln_revenue']

def read_hourly(modelName_base,modelName_EB,output_names,time_range_h,path):
    df_dict = {}

    for name in output_names:

        df_dict[f'base_{name}'] = pd.read_csv(path+f'df_{name}_year{modelName_base}.csv',index_col=0).set_index(time_range_h)
        df_dict[f'EB_{name}'] = pd.read_csv(path+f'df_{name}_year{modelName_EB}.csv',index_col=0).set_index(time_range_h)

    return df_dict

# Averages
def compute_averages(df_dict):

    # Weighted average of market clearing price
    if ('Network' in modelName_base):
        zone1 = ['grid_1','consumer_1','consumer_2','consumer_22','consumer_23']
        zone2 = np.setdiff1d(list(df_dict['base_uniform_price'].columns),zone1)

        zone1_weighted_base = df_dict['base_uniform_price'][zone1]*df_dict['base_Gn'][zone1] + df_dict['base_uniform_price'][zone1]*df_dict['base_Ln'][zone1]
        zone2_weighted_base = df_dict['base_uniform_price'][zone2]*df_dict['base_Gn'][zone2] + df_dict['base_uniform_price'][zone2]*df_dict['base_Ln'][zone2]

        zone1_weighted_EB = df_dict['EB_uniform_price'][zone1]*df_dict['EB_Gn'][zone1] + df_dict['EB_uniform_price'][zone1]*df_dict['EB_Ln'][zone1]
        zone2_weighted_EB = df_dict['EB_uniform_price'][zone2]*df_dict['EB_Gn'][zone2] + df_dict['EB_uniform_price'][zone2]*df_dict['EB_Ln'][zone2]

        Z1_tot_Q_base = df_dict['base_Gn'][zone1].sum(axis=1)+df_dict['base_Ln'][zone1].sum(axis=1)
        Z2_tot_Q_base = df_dict['base_Gn'][zone2].sum(axis=1)+df_dict['base_Ln'][zone2].sum(axis=1)

        Z1_tot_Q_EB = df_dict['EB_Gn'][zone1].sum(axis=1)+df_dict['EB_Ln'][zone1].sum(axis=1)
        Z2_tot_Q_EB = df_dict['EB_Gn'][zone2].sum(axis=1)+df_dict['EB_Ln'][zone2].sum(axis=1)

        row_sumZ1_base = zone1_weighted_base.sum(axis=1)
        row_sumZ2_base = zone2_weighted_base.sum(axis=1)

        row_sumZ1_EB = zone1_weighted_EB.sum(axis=1)
        row_sumZ2_EB = zone2_weighted_EB.sum(axis=1)

        df_dict['base_weighted_price'] = (row_sumZ1_base + row_sumZ2_base)/(Z1_tot_Q_base + Z2_tot_Q_base)
        df_dict['EB_weighted_price'] = (row_sumZ1_EB + row_sumZ2_EB)/(Z1_tot_Q_EB + Z2_tot_Q_EB)
    
    df_dict_avg = {}

    # 24 hour averages
    for key in df_dict.keys():
        if ('uniform_price' in key) & ('Network' in modelName_base): # zone 1: consumer 1, 2, 22, 23 (Grid); zone 2: rest (SM)
            df_dict_avg[f'{key}_zone1_davg'] = df_dict[key].grid_1.groupby(pd.Grouper(freq='24H')).mean() # this has to be changed when there is more than one grid
            df_dict_avg[f'{key}_zone2_davg'] = df_dict[key].sm_1.groupby(pd.Grouper(freq='24H')).mean() # this has to be changed when there is more than one SM
        else: df_dict_avg[f'{key}_davg'] = df_dict[key].groupby(pd.Grouper(freq='24H')).mean()

    # Weekly Averages
    for key in df_dict.keys():
        if ('uniform_price' in key) & ('Network' in modelName_base): # zone 1: consumer 1, 2, 22, 23 (Grid); zone 2: rest (SM)
            df_dict_avg[f'{key}_zone1_wavg'] = df_dict[key].grid_1.groupby(pd.Grouper(freq='1W')).mean() # this has to be changed when there is more than one grid
            df_dict_avg[f'{key}_zone2_wavg'] = df_dict[key].sm_1.groupby(pd.Grouper(freq='1W')).mean() # this has to be changed when there is more than one SM
        else: df_dict_avg[f'{key}_wavg'] = df_dict[key].groupby(pd.Grouper(freq='1W')).mean()
    
    # Monthly averages
    for key in df_dict.keys():
        if ('uniform_price' in key) & ('Network' in modelName_base): # zone 1: consumer 1, 2, 22, 23 (Grid); zone 2: rest (SM)
            df_dict_avg[f'{key}_zone1_mavg'] = df_dict[key].grid_1.groupby(pd.Grouper(freq='1M')).mean() # this has to be changed when there is more than one grid
            df_dict_avg[f'{key}_zone2_mavg'] = df_dict[key].sm_1.groupby(pd.Grouper(freq='1M')).mean() # this has to be changed when there is more than one SM
        else: df_dict_avg[f'{key}_mavg'] = df_dict[key].groupby(pd.Grouper(freq='1M')).mean()

    
    return df_dict_avg

df_dict = read_hourly(modelName_base,modelName_EB,output_names,time_range_h,path)
df_dict_avg = compute_averages(df_dict)

#%%

# plot the averages
if 'Network' in modelName_base:
    fig, axes = plt.subplots(2,1,sharex=True)
    df_dict_avg['base_uniform_price_zone1_mavg'].plot(ax=axes[0])
    df_dict_avg['base_uniform_price_zone2_mavg'].plot(ax=axes[0])
    axes[0].set_title('Montly Average Market Clearing Price Base')
    axes[0].set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
    axes[0].legend(['Zone 1', 'Zone 2'])

    df_dict_avg['EB_uniform_price_zone1_mavg'].plot(ax=axes[1])
    df_dict_avg['EB_uniform_price_zone2_mavg'].plot(ax=axes[1])
    axes[1].set_title('Monthly Average Market Clearing Price EB')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
    axes[1].legend(['Zone 1', 'Zone 2'])
else:
    fig, axes = plt.subplots(1,1,sharex=False)
    df_dict_avg['base_uniform_price_mavg'].plot(ax=axes)
    df_dict_avg['EB_uniform_price_mavg'].plot(ax=axes)

    axes.set_title('Monthly Average Market Clearing Price')
    axes.set_xlabel('Time')
    axes.set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
    axes.legend(['Base','EB'])
#plt.show()

# Plot the averages of each hour 
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='H')
flex_havg = df_dict['EB_Ln'].sm_1.loc[reduced_time].groupby(df_dict['EB_Ln'].sm_1.loc[reduced_time].index.hour).mean()
sm_consumption_havg = df_dict['base_Ln'].sm_1.loc[reduced_time].groupby(df_dict['base_Ln'].loc[reduced_time].index.hour).mean()
grid_price_havg = grid_price.loc[reduced_time].groupby(grid_price.loc[reduced_time].index.hour).mean()

cost = price_el_hourly/COP
cost_havg = cost.loc[reduced_time].groupby(cost.loc[reduced_time].index.hour).mean()

fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(flex_havg,alpha=0.5,linestyle='-', marker='o',markersize=2)
sm_consumption_havg.plot(ax=ax[0],alpha=0.3,linestyle='-', marker='o',markersize=2)
ax[0].set_title(f'SM Consumption Hourly Average: {reduced_time[0].strftime("%Y-%m-%d")} ~ {reduced_time[-1].strftime("%Y-%m-%d")}')
ax[0].set_ylabel('kWh')
ax[0].legend(['Energy Budget','Base'])

# plot the costs of supermarket and grid
ax[1].plot(cost_havg,color='orange')
ax[1].tick_params(axis ='y', labelcolor = 'orange')

ax2 = ax[1].twinx()
ax2.plot(cost_havg.index,grid_price_havg,color='blue')
ax2.tick_params(axis ='y', labelcolor = 'blue')

ax[1].set_xlabel('Hour')
ax[1].set_ylabel('SM Cost Euro [{}]'.format(unicodedata.lookup("EURO SIGN")),color='orange')
ax2.set_ylabel('Grid Cost Euro [{}]'.format(unicodedata.lookup("EURO SIGN")),color='blue')

# Plot the weekly average of settlements 
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='W')
cost_wavg = cost.groupby(pd.Grouper(freq='1W')).mean()

df_settlement_year_EB_wavg = df_dict_avg['EB_settlement_wavg']
df_settlement_year_base_wavg = df_dict_avg['base_settlement_wavg']
sm_settlement_EB_wavg_mean = df_settlement_year_EB_wavg.loc[reduced_time].sm_1.mean()
sm_settlement_base_wavg_mean = df_settlement_year_base_wavg.loc[reduced_time].sm_1.mean()

df_Gn_revenue_year_EB_wavg = df_dict_avg['EB_Gn_revenue_wavg']
df_Gn_revenue_year_base_wavg = df_dict_avg['base_Gn_revenue_wavg']

df_Ln_revenue_year_EB_wavg = df_dict_avg['EB_Ln_revenue_wavg']
df_Ln_revenue_year_base_wavg = df_dict_avg['base_Ln_revenue_wavg']

if 'Network' in modelName_base:
    df_uniform_price_EB_wavg = df_dict_avg['EB_weighted_price_wavg']
    df_uniform_price_base_wavg = df_dict_avg['base_weighted_price_wavg']
else: 
    df_uniform_price_EB_wavg = df_dict_avg['EB_uniform_price_wavg']
    df_uniform_price_base_wavg = df_dict_avg['base_uniform_price_wavg']

fig, ax = plt.subplots(2,1,sharex=True)
df_settlement_year_base_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
df_settlement_year_EB_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
ax[0].axhline(y=sm_settlement_base_wavg_mean)
ax[0].axhline(y=sm_settlement_EB_wavg_mean,color='orange')
ax[0].legend(['Base','Energy Budget'])
ax[0].set_title('SM Weekly Average of Settlement')

df_uniform_price_base_wavg.loc[reduced_time].plot(ax=ax[1])
df_uniform_price_EB_wavg.loc[reduced_time].plot(ax=ax[1])
ax[1].set_title('Weekly Average of Market Clearing Price')
ax[1].legend(['Base','Energy Budget'])
plt.tight_layout()
#plt.show()

# SM and Grid production weekly average
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='1W')

if 'Network' in modelName_base:
    mrkt_mean_base = df_dict_avg['base_weighted_price_wavg'].loc[reduced_time].mean()[0]
    mrkt_mean_EB = df_dict_avg['EB_weighted_price_wavg'].loc[reduced_time].mean()[0]
else:
    mrkt_mean_base = df_dict_avg['base_uniform_price_wavg'].loc[reduced_time].mean()[0]
    mrkt_mean_EB = df_dict_avg['EB_uniform_price_wavg'].loc[reduced_time].mean()[0]

fig, ax = plt.subplots(3,1,sharex=True)

df_uniform_price_base_wavg.loc[reduced_time].plot(ax=ax[0])
df_uniform_price_EB_wavg.loc[reduced_time].plot(ax=ax[0])
    
ax[0].axhline(y=mrkt_mean_base)
ax[0].axhline(y=mrkt_mean_EB,color='orange')
ax[0].set_title('Weekly Average of Market Clearing Price')
ax[0].legend(['Base','Energy Budget'])

df_dict_avg['base_Gn_wavg'].loc[reduced_time,'sm_1'].plot(ax=ax[1])
df_dict_avg['EB_Gn_wavg'].loc[reduced_time,'sm_1'].plot(ax=ax[1])
ax[1].legend(['Base SM','EB SM'],loc='upper right')
ax[1].set_title('SM Production')

df_dict_avg['base_Gn_wavg'].loc[reduced_time,'grid_1'].plot(ax=ax[2])
df_dict_avg['EB_Gn_wavg'].loc[reduced_time,'grid_1'].plot(ax=ax[2])
ax[2].legend(['Base Grid', 'EB Grid'])
ax[2].set_title('Grid Production')
plt.tight_layout()
plt.show()

# SM Revenue Weekly Average
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='W')
fig, ax = plt.subplots(3,1,sharex=True)
df_Gn_revenue_year_base_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
df_Gn_revenue_year_EB_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
ax[0].legend(['Base','Energy Budget'])
ax[0].set_title('SM Weekly Average of Production Revenue')

df_Ln_revenue_year_base_wavg.loc[reduced_time].sm_1.plot(ax=ax[1])
df_Ln_revenue_year_EB_wavg.loc[reduced_time].sm_1.plot(ax=ax[1])
ax[1].legend(['Base','Energy Budget'])
ax[1].set_title('SM Weekly Average of Consumption Revenue')

df_uniform_price_base_wavg.loc[reduced_time].plot(ax=ax[2])
df_uniform_price_EB_wavg.loc[reduced_time].plot(ax=ax[2])
ax[2].set_title('Weekly Average of Market Clearing Price')
ax[2].legend(['Base','Energy Budget'])
plt.tight_layout()
#plt.show()

# Plot the weekly average of social welfare
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='1W')
df_sw_year_base_wavg = df_dict_avg['base_sw_wavg']*-1
df_sw_year_EB_wavg = df_dict_avg['EB_sw_wavg']*-1
cost_wavg = cost.groupby(pd.Grouper(freq='1W')).mean()

fig, ax = plt.subplots()
df_sw_year_base_wavg.loc[reduced_time].plot(ax=ax)
df_sw_year_EB_wavg.loc[reduced_time].plot(ax=ax)
ax.legend(['Base','Energy Budget'])
ax.set_title('Weekly Average Socical Welfare')

cost_wavg.loc[reduced_time].plot(ax=ax[1])
df_uniform_price_base_wavg.loc[reduced_time].plot(ax=ax[1])
df_uniform_price_EB_wavg.loc[reduced_time].plot(ax=ax[1])
ax[1].set_title('Weekly Average Market Clearing Price')
ax[1].legend(['Base','Energy Budget'])


# Consumption of supermarket and consumers
consumers_base = df_dict['base_Ln'].loc[:,'consumer_1':].groupby(pd.Grouper(freq='1W')).mean().sum(axis=1)
consumers_EB = df_dict['EB_Ln'].loc[:,'consumer_1':].groupby(pd.Grouper(freq='1W')).mean().sum(axis=1)
sm_consumption_base_wavg = df_dict['base_Ln'].sm_1.groupby(pd.Grouper(freq='1W')).mean()
sm_consumption_EB_wavg = df_dict['EB_Ln'].loc[:,'sm_1'].groupby(pd.Grouper(freq='1W')).mean()

fig, ax = plt.subplots(2,1,sharex=True)
sm_consumption_base_wavg.plot(ax=ax[0])
sm_consumption_EB_wavg.plot(ax=ax[0])
consumers_base.plot(ax=ax[1])
consumers_EB.plot(ax=ax[1])
ax[0].set_title('SM Consumption')
ax[0].legend(['SM_base','SM_EB'])
ax[1].legend(['Consumer_base','Consumer_EB'])
ax[1].set_title('Weekly Demand Consumption')
plt.show()
