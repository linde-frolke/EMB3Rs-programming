import pandas as pd
from market_module.DTU_case_study_new.save_tocsv import save_tocsv, save_topickle, load_frompickle, CaseStudyData
from market_module.DTU_case_study_new.prep_inputs import prep_inputs
import matplotlib.pyplot as plt

# get inputs
one_year_idx, nr_h, agent_ids, g_max, l_max, cost, utility, time_range = prep_inputs()

## value of flexibility, pool ---------------------------------------------------
pool = load_frompickle("Pool_base")
pool_EB = load_frompickle("Pool_EB")

#pool_EB.Ln.head(len(time_range)).set_index(time_range)
#pool.Ln.plot()
#plt.show()

plt.plot(time_range, pool.Ln.sum(axis=1), label="pool")
plt.plot(time_range, pool_EB.Ln.sum(axis=1).head(len(time_range)), label="pool EB")
plt.show()

# plot the difference 
plt.plot(time_range, pool.Ln.sum(axis=1) -  pool_EB.Ln.sum(axis=1).head(len(time_range)))
plt.show()
