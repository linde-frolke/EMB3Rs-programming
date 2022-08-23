import pandas as pd
import numpy as np
from market_module.DTU_case_study_new.save_tocsv import save_tocsv, save_topickle, load_frompickle, CaseStudyData
from market_module.DTU_case_study_new.prep_inputs import prep_inputs
import matplotlib.pyplot as plt

# figure settings --------------------------------------------------------------
# to save figures
folder = "/home/linde/Documents/2019PhD/EMB3Rs/WP4_case_study/results/figs/"
color_sm = "blue"
color_res = "yellow"
figtype = ".png"

# get inputs
one_year_idx, nr_h, agent_ids, g_max, l_max, cost, utility, time_range = prep_inputs()

# get results 
pool = load_frompickle("Pool_base")
pool_EB = load_frompickle("Pool_EB")
p2p = load_frompickle("P2P_base")
# get insight in inputs 
df_lmax = pd.DataFrame(l_max, index = pool.Ln.index, columns=pool.Ln.columns)
mnts = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
barWidth = 0.4
br1 = np.arange(12)
br2 = [x + barWidth for x in br1]
plt.bar(br1, height=df_lmax.iloc[:,1:2].sum(axis=1).groupby(df_lmax.index.month).sum() / 10**3, label="supermarket",  width = barWidth)
plt.bar(br2, height=df_lmax.iloc[:,2:].sum(axis=1).groupby(df_lmax.index.month).sum() / 10**3, label="residential",  width = barWidth)
plt.xlabel("month")
plt.ylabel("total monthly load [MWh]")
plt.xticks([r + barWidth for r in range(12)], mnts)
plt.legend()
plt.savefig(folder + "inputs_Ln_monthly" + figtype)
plt.show()



## value of flexibility, pool ---------------------------------------------------


# how does flexibility affect the scheduled load? --------
# plot the difference in load profile for the consumers
plt.plot(pool.Ln.iloc[:, 2:].sum(axis=1).groupby(pool.Ln.index.hour).mean(), label="w/o Energy Budget")
plt.plot(pool_EB.Ln.iloc[:, 2:].sum(axis=1).groupby(pool_EB.Ln.index.hour).mean(), label="w/  Energy Budget")
# plt.plot(2*df_lmax.iloc[:, 2:].sum(axis=1).groupby(pool.Ln.index.hour).mean(), label="average max load", color="gray", linestyle="dotted")
plt.legend()
plt.xlabel("time of day [h]")
plt.ylabel("average residential load [kWh]")
plt.savefig(folder + "pool_compare_Ln_EB.pdf")
#plt.plot(df_lmax.iloc[:, 2:].sum(axis=1).groupby(pool.Ln.index.hour).mean())
plt.show()

# how does flexibility affect the market price?
# monthly average market clearing price with respect the two models, i.e. base and energy budget
mnts = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
barWidth = 0.4
br1 = np.arange(12)
br2 = [x + barWidth for x in br1]
unit = 10**3
plt.bar(br1, height=pool.price.groupby(pool.price.index.month).mean().sum(axis=1) / unit, label="w/o Energy Budget",  width = barWidth)
plt.bar(br2, height=pool_EB.price.groupby(pool.price.index.month).mean().sum(axis=1) / unit, label="w/  Energy Budget", width = barWidth)
plt.xlabel("month")
plt.ylabel("avg market price [€/MWh]")
plt.xticks([r + barWidth for r in range(12)], mnts)
plt.legend()
plt.savefig(folder + "pool_compare_price_EB" + figtype)
plt.show()

# how does flexibility affect the schedules of the grid and supermarket?
# montly total Gn with respect the two models, i.e. base and energy budget
unit = 10**3
plt.plot(br1, pool.Gn.loc[:,"grid_1"].groupby(pool.price.index.month).sum() / unit, label="grid w/o Energy Budget")
plt.plot(br1, pool.Gn.loc[:,"sm_1"].groupby(pool.price.index.month).sum() / unit, label="supermarket w/o Energy Budget")
plt.plot(br1, pool_EB.Gn.loc[:,"grid_1"].groupby(pool.price.index.month).sum() / unit, label="grid w/  Energy Budget", linestyle="dotted")
plt.plot(br1, pool_EB.Gn.loc[:,"sm_1"].groupby(pool.price.index.month).sum() / unit, label="supermarket w/  Energy Budget", linestyle="dotted")
plt.xlabel("month")
plt.ylabel("scheduled generation [MWh]")
plt.xticks([r + barWidth for r in range(12)], mnts)
plt.legend()
plt.savefig(folder + "pool_compare_Gn_monthly_EB" + figtype)
plt.show()

# social welfare
"TODO"

# check that p2p base is same as pool -- it is. 
plt.plot(pool.Gn.loc[:, "grid_1"] - p2p.Gn.loc[:, "grid_1"])
plt.show()
plt.plot(pool.Gn - p2p.Gn) # insignificant differences
plt.show()

plt.plot(pool.Ln - p2p.Ln)
plt.show()
