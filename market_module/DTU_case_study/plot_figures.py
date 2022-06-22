from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import unicodedata
import numpy as np
import seaborn
from market_module.DTU_case_study.load_data import load_data

path = 'C://Users//hyung//Documents//GitHub//EMB3Rs-programming//market_module//DTU_case_study//Data//'

# Load flexibility
Year = 2018
Month = 4
Day = 14
tot_h = 365*24
COP = 2.5 

consumption_data, price_el_hourly, grid_price = load_data(tot_h,Year,Month,Day)
sm_consumption = consumption_data.loc[:,'SM_consumption']
flex = pd.read_csv(path+'flexibility.csv',usecols=[1]).set_index(sm_consumption.index)

# Base model
time_range_d = pd.date_range(start=consumption_data.index[0], end=consumption_data.index[-1],freq='D')
time_range_m = pd.date_range(start=consumption_data.index[0], end=consumption_data.index[-1]+timedelta(days=30),freq='M')

df_uniform_price_base = pd.read_csv(path+'df_uniform_price_base.csv',index_col=0).set_index(time_range_d)
df_Pn_year_base = pd.read_csv(path+'df_Pn_year_base.csv',index_col=0).set_index(time_range_d)
df_Gn_year_base = pd.read_csv(path+'df_Gn_year_base.csv',index_col=0).set_index(time_range_d)
df_Ln_year_base = pd.read_csv(path+'df_Ln_year_base.csv',index_col=0).set_index(time_range_d)

df_sw_year_base = pd.read_csv(path+'df_sw_year_base.csv',index_col=0).set_index(time_range_d)
df_settlement_year_base = pd.read_csv(path+'df_settlement_year_base.csv',index_col=0).set_index(time_range_d)

df_uniform_mavg_base = pd.read_csv(path+'df_uniform_price_mavg_base.csv',index_col=0).set_index(time_range_m)
df_Pn_year_mavg_base = pd.read_csv(path+'df_Pn_year_mavg_base.csv',index_col=0).set_index(time_range_m)
df_Ln_year_mavg_base = pd.read_csv(path+'df_Ln_year_mavg_base.csv',index_col=0).set_index(time_range_m)
df_sw_year_mavg_base = pd.read_csv(path+'df_sw_year_mavg_base.csv',index_col=0).set_index(time_range_m)
df_settlement_mavg_base = pd.read_csv(path+'df_settlement_year_mavg_base.csv',index_col=0).set_index(time_range_m)


# Energy Budget
df_uniform_price_EB = pd.read_csv(path+'df_uniform_price.csv',index_col=0).set_index(time_range_d)
df_Pn_year_EB = pd.read_csv(path+'df_Pn_year.csv',index_col=0).set_index(time_range_d)
df_Gn_year_EB = pd.read_csv(path+'df_Gn_year.csv',index_col=0).set_index(time_range_d)
df_Ln_year_EB = pd.read_csv(path+'df_Ln_year.csv',index_col=0).set_index(time_range_d)

df_sw_year_EB = pd.read_csv(path+'df_sw_year.csv',index_col=0).set_index(time_range_d)
df_settlement_year_EB = pd.read_csv(path+'df_settlement_year.csv',index_col=0).set_index(time_range_d)

df_uniform_mavg_EB = pd.read_csv(path+'df_uniform_price_mavg.csv',index_col=0).set_index(time_range_m)
df_Pn_year_mavg_EB = pd.read_csv(path+'df_Pn_year_mavg.csv',index_col=0).set_index(time_range_m)
df_Ln_year_mavg_EB = pd.read_csv(path+'df_Ln_year_mavg.csv',index_col=0).set_index(time_range_m)
df_sw_year_mavg_EB = pd.read_csv(path+'df_sw_year_mavg.csv',index_col=0).set_index(time_range_m)
df_settlement_mavg_EB = pd.read_csv(path+'df_settlement_year_mavg.csv',index_col=0).set_index(time_range_m)


# plot the averages
fig, axes = plt.subplots(2,1,sharex=True)
#axes.plot(df_uniform_mavg_EB)
#axes.plot(df_uniform_mavg_base)
flex_mavg = flex.groupby(pd.Grouper(freq='1M')).mean()

df_uniform_mavg_EB.plot(ax=axes[0])
df_uniform_mavg_base.plot(ax=axes[0])
axes[0].set_title('Montly Average Market Clearing Price')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
axes[0].legend(['Energy Budget','Base'])

flex_mavg.plot(ax=axes[1],label='Energy Budget Consumption')
axes[1].legend(['Flexiblity Consumption Monthly Average'])
plt.show()

# Monthly average of consumption flexiblity
# They are exactly the same, because it evens out per day in 24hours.
fig,ax = plt.subplots()
df_Ln_year_mavg_EB.sm_1.plot(ax=ax)
df_Ln_year_mavg_base.sm_1.plot(ax=ax)
plt.legend(['Energy Budget','Base'])
plt.show()

# Plot the averages of each hour 
# The Energy Budget curve seems to follow the electricity curve more than the fixed consumption
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='H')
flex_havg = flex.loc[reduced_time].groupby(flex.loc[reduced_time].index.hour).mean()
sm_consumption_havg = sm_consumption[reduced_time].groupby(sm_consumption[reduced_time].index.hour).mean()

cost = price_el_hourly/COP
cost_havg = cost.loc[reduced_time].groupby(cost.loc[reduced_time].index.hour).mean()
fig,ax = plt.subplots(2,1,sharex=True)
ax[0].plot(flex_havg,alpha=0.5,label='Actual',linestyle='-', marker='o',markersize=2)
sm_consumption_havg.plot(ax=ax[0],alpha=0.3,label='Flexible',linestyle='-', marker='o',markersize=2)
ax[0].set_title(f'Consumption Hourly Average: {reduced_time[0].strftime("%Y-%m-%d")} ~ {reduced_time[-1].strftime("%Y-%m-%d")}')
ax[0].set_ylabel('kWh')
ax[0].legend(['Energy Budget','Base'])
ax[1].plot(cost_havg)
ax[1].set_xlabel('Hour')
ax[1].set_ylabel('Euro [{}]'.format(unicodedata.lookup("EURO SIGN")))
ax[1].legend(['Electricity Cost'])
plt.show()

# Plot the settlements (cleared by day)
# It is expected the settlement would closely follow market clearing price as settlement is directly calculated from it
# The constant in the two different scenarios is that the electricity prices are fixed, from the EB we see it follows the el_dep
# more than the base model
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='W')
cost_wavg = cost.groupby(pd.Grouper(freq='1W')).mean()
df_settlement_year_EB_wavg = df_settlement_year_EB.groupby(pd.Grouper(freq='1W')).mean()
df_settlement_year_base_wavg = df_settlement_year_base.groupby(pd.Grouper(freq='1W')).mean()
df_uniform_price_base_wavg = df_uniform_price_base.groupby(pd.Grouper(freq='1W')).mean()
df_uniform_price_EB_wavg = df_uniform_price_EB.groupby(pd.Grouper(freq='1W')).mean()

fig, ax = plt.subplots(3,1,sharex=True)
df_settlement_year_base_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
df_settlement_year_EB_wavg.loc[reduced_time].sm_1.plot(ax=ax[0])
ax[0].legend(['Base','Energy Budget'])
ax[0].set_title('Weekly Average of Settlement')
cost_wavg.loc[reduced_time].plot(ax=ax[1])
ax[1].set_title('Weekly Average of Electricity Prices')
df_uniform_price_base_wavg.loc[reduced_time].plot(ax=ax[2])
df_uniform_price_EB_wavg.loc[reduced_time].plot(ax=ax[2])
ax[2].set_title('Weekly Average of Market Clearing Price')
ax[2].legend(['Base','Energy Budget'])
plt.tight_layout()
plt.show()


# Plot the social welfare
# We clearly see that with EB the demand consumes less therefore has a higher sw and settlement  
reduced_time = pd.date_range(start='2018-11',end='2019-4-13',freq='1W')
df_sw_year_base_wavg = df_sw_year_base.groupby(pd.Grouper(freq='1W')).mean()*-1
df_sw_year_EB_wavg = df_sw_year_EB.groupby(pd.Grouper(freq='1W')).mean()*-1
cost_wavg = cost.groupby(pd.Grouper(freq='1W')).mean()
fig, ax = plt.subplots(2,1,sharex=True)
df_sw_year_base_wavg.loc[reduced_time].plot(ax=ax[0])
df_sw_year_EB_wavg.loc[reduced_time].plot(ax=ax[0])
ax[0].legend(['Base','Energy Budget'])
ax[0].set_title('Weekly Average Socical Welfare')
cost_wavg.loc[reduced_time].plot(ax=ax[1])
ax[1].set_title('Weekly Average Electricity Price')
plt.show()