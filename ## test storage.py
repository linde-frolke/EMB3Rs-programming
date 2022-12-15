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

# input_data.keys()
# input_data["user"].keys()
# input_data["teo-module"]["ex_capacities"]
# input_data["teo-module"]["AccumulatedNewStorageCapacity"]
# ProductionByTechnologyAnnual = pd.DataFrame(input_data["teo-module"].get("ProductionByTechnologyMM", input_data["teo-module"]["ProductionByTechnology"]))
# ProductionByTechnologyAnnual[ProductionByTechnologyAnnual["TECHNOLOGY"] == "gridspecificbioboiler"].sort_values(by=["TIMESLICE"]).filter(items=["VALUE"]).values

#pd.DataFrame(input_data["teo-module"].get("ProductionByTechnologyMM", input_data["teo-module"]["ProductionByTechnology"]))

from market_module.long_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs #, run_longterm_market
from market_module.long_term.market_functions.run_longterm_market import run_longterm_market

input_data["user"]["start_datetime"] = "2023-01-01"
#input_data["recurrence"] = 1
#input_data["horizon_basis"] = "months"
# check whether the correct inputs are created
input_dict = convert_user_and_module_inputs(input_data)

len(input_dict["gmax"])
# input_dict["storage_capacity"]
# input_dict["storage_name"]

# (np.array(input_dict["storage_capacity"]).shape)

# 
output = run_longterm_market(input_dict=input_dict)
output.keys()

pd.DataFrame(output["Bn"]).iloc[1,:].sum() #).iloc([0,:])
(pd.DataFrame(output["Pn"]).iloc[1,:]).sum()
pd.DataFrame(output["En"]).iloc[:100,0].plot() #.rolling(window=6).sum().plot()
plt.show()

#output.keys()
output["optimal"]

## check if works with 1 storage only 
input_data_1stor = input_data.copy()
test = pd.DataFrame(input_data_1stor["teo-module"]["AccumulatedNewStorageCapacity"])
test
test[test.TECHNOLOGY == "dhn"]
input_data_1stor["teo-module"]["AccumulatedNewStorageCapacity"] = [input_data_1stor["teo-module"]["AccumulatedNewStorageCapacity"][0]]
input_dict_1stor = convert_user_and_module_inputs(input_data_1stor)
output_1stor = run_longterm_market(input_dict=input_dict_1stor)

## check if still works without storage ---------------------
input_data_nostor = input_data.copy()
input_data_nostor["teo-module"]["AccumulatedNewStorageCapacity"] = []
input_dict_nostor = convert_user_and_module_inputs(input_data_nostor)

output_nostor = run_longterm_market(input_dict=input_dict_nostor)
output_nostor.keys()
pd.DataFrame(output_nostor["Gn"]).sum(axis=1).plot()
pd.DataFrame(output_nostor["Ln"]).sum(axis=1).plot()
plt.show()
(pd.DataFrame(output_nostor["Gn"]).sum(axis=1) - pd.DataFrame(output_nostor["Ln"]).sum(axis=1)).plot()
plt.show()
# sum(pd.DataFrame(output["Pn"]).iloc[1,:])
# sum(pd.DataFrame(output["Pn"]).iloc[2,:])
# sum(pd.DataFrame(output["Pn"]).iloc[3,:])

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