import pandas as pd
from market_module.DTU_case_study_new.save_tocsv import save_tocsv, save_topickle, load_frompickle, CaseStudyData
import matplotlib.pyplot as plt

## value of flexibility, pool ---------------------------------------------------
pool = load_frompickle("Pool_base")
pool_EB = load_frompickle("Pool_EB")

pool_EB.Ln.index
pool_data.Ln.plot()
plt.show()