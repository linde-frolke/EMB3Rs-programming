"""
test function for long term market, centralized
"""

import os
import numpy as np
import pandas as pd
import json

cwd = os.getcwd()

# import own modules
from market_module.long_term.market_functions.run_longterm_market import run_longterm_market

# read file --------------
fn = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/AccumulatedNewCapacity.json"
f = open(fn, "r")
teo_vals = json.load(f)
teo_output = {"AccumulatedNewCapacity": teo_vals, "AnnualVariableOperatingCost": teo_vals, 
                "ProductionByTechnologyAnnual": teo_vals}

# 
AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])

# user inputs ---------------------------
year = 2000 # must be one of the years that the TEO simulates for
year = AccumulatedNewCapacity.YEAR.min()

# extract needed values ------------------
# select year
AccumulatedNewCapacity = AccumulatedNewCapacity[AccumulatedNewCapacity.YEAR == year]
# select nonzero generators
AccumulatedNewCapacity = AccumulatedNewCapacity[AccumulatedNewCapacity.VALUE > 0]

# extract source names
source_names = AccumulatedNewCapacity.TECHNOLOGY.unique()
# give them a number 
source_names = source_names[pd.notna(source_names)]

# teo_split_name = "Source_SourceID_StreamID_HeatExchanger".split("_")


## get stuff from CF ----------------------------------------
# sources

fncf = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/convert_sources_output.jsonc"
f = open(fncf, "r")
#cf_vals = pd.read_json(path_or_buf=fncf, )
all_sources_info = json.load(f)
all_sources_info = all_sources_info["all_sources_info"]

nr_of_sources = len(all_sources_info)

i  = 0
source_ids = []
source_locs = []

all_sources_info[i]

for i in range(nr_of_sources):
    source_ids += [all_sources_info[i]["source_id"]]
    source_locs += [all_sources_info[i]["location"]]

type(all_sources_info[0])


# sinks 
fncf2 = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/convert_sinks.jsonc"
f = open(fncf2, "r")
#cf_vals = pd.read_json(path_or_buf=fncf, )
all_sinks_info = json.load(f)
all_sinks_info = all_sinks_info["all_sinks_info"]["sinks"]

nr_of_sinks = len(all_sinks_info)

# get some sink info 
sink_ids = []
sink_locs = []
map_sink_streams = []

for i in range(nr_of_sinks):
    sink_ids += [all_sinks_info[i]["sink_id"]]
    sink_locs += [all_sinks_info[i]["location"]]
    map_sink_streams += [[streams["stream_id"] for streams in all_sinks_info[i]["streams"]]]

# get some stream info 
all_stream_ids = [y for x in map_sink_streams for y in x]
nr_of_streams = len(all_stream_ids)
timesteps = len(all_sinks_info[0]["streams"][0]["hourly_stream_capacity"])

lmax = np.zeros((nr_of_streams, timesteps))
counter = 0 
for j in range(nr_of_sinks):
    for i in range(len(map_sink_streams[j])):
        lmax[counter, :] = all_sinks_info[j]["streams"][i]["hourly_stream_capacity"]
        counter += 1


type(all_sources_info[0])


# need to get :
# sink ID
# teo_dem_factor
# teo_yearly_demand

## Combine them in 
agent_ids
gmax
lmax 
cost 

## how to do util?

