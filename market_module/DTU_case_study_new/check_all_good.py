# script for checking whether all is right

import pandas as pd
import numpy as np
from market_module.DTU_case_study_new.save_tocsv import save_tocsv, save_topickle, load_frompickle, CaseStudyData
from market_module.DTU_case_study_new.prep_inputs import prep_inputs
import matplotlib.pyplot as plt

# get inputs
one_year_idx, nr_h, agent_ids, g_max, l_max, cost, utility, time_range = prep_inputs()

## difference between base and EB load?? ---------------------------------------------------
pool = load_frompickle("Pool_base")
pool_EB = load_frompickle("Pool_EB")

plt.plot(time_range, pool.Ln.sum(axis=1), label="pool")
plt.plot(time_range, pool_EB.Ln.sum(axis=1).head(len(time_range)), label="pool EB")
plt.show()

# plot the difference 
daily_load = pool.Ln.groupby(pool.Ln.index.date).sum().sum(axis=1)
daily_load_EB = pool_EB.Ln.groupby(pool_EB.Ln.index.date).sum().sum(axis=1)

plt.plot((daily_load - daily_load_EB))
plt.show()

(daily_load != daily_load_EB).sum()
wrong_dates = daily_load.index[(daily_load - daily_load_EB).abs() > 0.1]

# check out those days 
start_date = pd.to_datetime(wrong_dates[0])
end_date = pd.to_datetime(wrong_dates[0]) + pd.Timedelta(days=1)

time_frame = (pool.Ln.index >= start_date) & (pool.Ln.index < end_date)

# check that budget is not the same on that day
pool.Ln[time_frame].sum() - pool_EB.Ln[time_frame].sum()

plt.plot(pool.Ln[time_frame].sum(axis=1), label="no EB")
plt.plot(pool_EB.Ln[time_frame].sum(axis=1), label="EB")
df_lmax = pd.DataFrame(l_max).set_index(pool.Ln.index)
plt.plot(df_lmax[time_frame].sum(axis=1), label="lmax")
plt.legend()
plt.show()

df_cost = pd.DataFrame(cost).set_index(pool.Ln.index)
df_util = pd.DataFrame(utility).set_index(pool.Ln.index)
df_cost[time_frame]
df_util[time_frame]
df_gmax = pd.DataFrame(g_max).set_index(pool.Ln.index)

# plot the difference between lmax and pool
daily_load = pool.Ln.groupby(pool.Ln.index.date).sum().sum(axis=1)
daily_load_EB = pool_EB.Ln.groupby(pool_EB.Ln.index.date).sum().sum(axis=1)

plt.plot(pool.Ln.sum(axis=1), label="Ln no EB")
plt.plot(df_lmax.sum(axis=1), label="lmax")
plt.plot(df_gmax.sum(axis=1), label="gmax")
plt.legend()
plt.show()

plt.plot(pool.Ln.sum(axis=1) - df_lmax.sum(axis=1))
plt.show()

df_lmax.sum(axis=1).max()