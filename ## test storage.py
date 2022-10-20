## test storage

import numpy as np
import pandas as pd
# import xlrd
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import datetime
import json 
from math import ceil
import matplotlib.pyplot as plt

f1 = open("/home/linde/Documents/2019PhD/EMB3Rs/module_integration/UoW-withstorage-market-module-long-term-input.json")
input_data = json.load(f1)

input_data.keys()
input_data["user"]
input_data["teo-module"]["AccumulatedNewStorageCapacity"]

from market_module.long_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs #, run_longterm_market
from market_module.long_term.market_functions.run_longterm_market import run_longterm_market

input_data["user"]["start_datetime"] = "2023-01-01"
#input_data["recurrence"] = 1
#input_data["horizon_basis"] = "months"
# check whether the correct inputs are created
input_dict = convert_user_and_module_inputs(input_data)

# input_dict["storage_capacity"]
# input_dict["storage_name"]

# (np.array(input_dict["storage_capacity"]).shape)

# 
output = run_longterm_market(input_dict=input_dict)
output.keys()

pd.DataFrame(output["Bn"]).iloc[1,:].sum() #).iloc([0,:])

pd.DataFrame(output["En"]).iloc[0:49,:].plot()
plt.show()

#output.keys()
output["optimal"]
(pd.DataFrame(output["Pn"]).iloc[1,:]).sum()
# sum(pd.DataFrame(output["Pn"]).iloc[1,:])
# sum(pd.DataFrame(output["Pn"]).iloc[2,:])
# sum(pd.DataFrame(output["Pn"]).iloc[3,:])

# %import cvxpy
# %print(cvxpy.__version__)

from market_module.long_term.datastructures.inputstructs import AgentData, MarketSettings

settings = MarketSettings(product_diff=input_dict['prod_diff_option'], market_design=input_dict['md'],
                            horizon_basis=input_dict['horizon_basis'],
                            recurrence=input_dict['recurrence'], data_profile=input_dict['data_profile'],
                            ydr=input_dict['yearly_demand_rate'], start_datetime=input_dict['start_datetime'])


agent_data = AgentData(settings=settings, name=input_dict['agent_ids'],
                        #a_type=input_dict['agent_types'],
                        gmax=input_dict['gmax'],
                        lmax=input_dict['lmax'],
                        cost=input_dict['cost'], util=input_dict['util'], co2=input_dict['co2_emissions'], 
                        storage_name= input_dict["storage_name"], storage_capacity= input_dict["storage_capacity"]
                        )

agent_data.gmax


agent_data.gmax.to_numpy()[range(0,10),:]

type(agent_data.storage_capacity)

type(input_dict["storage_capacity"])
agent_data.storage_name
agent_data.storage_capacity