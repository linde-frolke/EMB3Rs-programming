"""
test function for long term market, centralized
"""

import os
import numpy as np
import pandas as pd
import json
from ast import literal_eval

cwd = os.getcwd()

# import own modules
from market_module.long_term.market_functions.run_longterm_market import run_longterm_market

# read gis file
fn_gis = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/optimize_network.output.json"
f = open(fn_gis, "r")
gis_vals = json.load(f)["output_data"]

gis_vals.keys()

gis_data = gis_vals["res_sources_sinks"]
gis_vals

df = pd.DataFrame(data=gis_data)
df["from_to"] = [literal_eval(x) for x in df["from_to"]]
type(df["from_to"][0])

nodes = [x["osmid"] for x in  gis_vals["network_solution_nodes"]]
edges = pd.DataFrame(gis_vals["network_solution_edges"])

from_to = [(edges.loc[i, "from"], edges.loc[i, "to"]) for i in range(len(edges))]

# read file --------------
fn = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/AccumulatedNewCapacity.json"
f = open(fn, "r")
teo_vals = json.load(f)
teo_output = {"AccumulatedNewCapacity": teo_vals, "AnnualVariableOperatingCost": teo_vals, 
                "ProductionByTechnologyAnnual": teo_vals}

# 
AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])

# user inputs ---------------------------
start_datetime = "31-01-2000"


# extract day month year
day, month, year = [int(x) for x in start_datetime.split("-")]
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
# agent_ids
# gmax
# lmax 
# cost 

## how to do util?



#
input_dict = {#'sim_name': 'test_pool',
                  'md': 'pool',  # other options are  'p2p' or 'community'
                  'nr_of_hours': 12,
                  'offer_type': 'simple',
                  'prod_diff': 'noPref',
                  'network': 'none',
                  'el_dependent': 'false',  # can be false or true
                  'el_price': 'none',
                  'agent_ids': ["prosumer_1",
                                "prosumer_2", "consumer_1", "producer_1"],
                  'agent_types': ["prosumer", "prosumer", "consumer", "producer"],
                  'objective': 'none',  # objective for community
                  'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'},
                  'gmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  'gmax': [[1, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3], [0, 0, 0, 0], [1, 1, 0, 1],
                           [2, 3, 0, 1], [4, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3],
                           [0, 0, 0, 0], [1, 1, 0, 1], [2, 3, 0, 1]],
                  'lmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  'lmax': [[2, 2, 1, 0], [2, 1, 0, 0], [1, 2, 1, 0], [3, 0, 2, 0], [1, 1, 4, 0],
                           [2, 3, 3, 0], [4, 2, 1, 0], [3, 4, 2, 0], [1, 5, 3, 0], [0, 0, 5, 0],
                           [1, 1, 3, 0], [2, 3, 1, 0]],
                  'cost': [[24, 25, 45, 30], [31, 24, 0, 24], [18, 19, 0, 32], [0, 0, 0, 0],
                           [20, 25, 0, 18], [25, 31, 0, 19], [24, 27, 0, 22], [32, 31, 0, 19],
                           [15, 25, 0, 31], [0, 0, 0, 0], [19, 20, 0, 21], [22, 33, 0, 17]],
                  'util': [[40, 42, 35, 0], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0],
                           [34, 44, 55, 0], [29, 33, 45, 0], [40, 55, 33, 0],
                           [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]],
                  'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                  'is_in_community': 'none',  # allowed values are 'none' or boolean array of size (nr_of_agents)
                  'block_offer': 'none',
                  'is_chp': 'none',  # allowed values are 'none' or a list with ids of agents that are CHPs
                  'chp_pars': 'none',
                  'gis_data': 
                        {'from_to': ['(0, 1)', '(1, 2)', '(1, 3)'],
                       'losses_total': [22969.228855, 24122.603833, 18138.588662],
                       'length': [1855.232413, 1989.471069, 1446.688900],
                       'total_costs': [1.848387e+06, 1.934302e+06, 1.488082e+06]},
                  'nodes' : ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"],
                  'edges' : [("producer_1","consumer_1"), ("producer_1","prosumer_1"),
                             ("prosumer_1","prosumer_2"), ]
                  }


tmp = np.array(input_dict["util"])
util = {}
for i in range(tmp.shape[1]):
    util[input_dict["agent_ids"][i]] = tmp[:, i].tolist()

util.pop("producer_1")
util


fn_gis = "/home/linde/Documents/2019PhD/EMB3Rs/EMB3Rs-programming/market_module/short_term/tests/optimize_network.output.json"
f = open(fn_gis, "r")
gis_vals = json.load(f)["output_data"]

gis_vals["res_sources_sinks"]

res_sources_sinks = pd.DataFrame(input_dict["gis_data"]).to_dict("records")

[gis_vals["network_solution_nodes"][i]["osmid"] for i in range(len(gis_vals["network_solution_nodes"]))]

gis_vals["network_solution_nodes"]

[{"osmid" : input_dict["nodes"][i]} for i in range(len(input_dict["nodes"]))]

gis_vals["network_solution_edges"]

[{"from" : input_dict["edges"][i][0], 
  "to" : input_dict["edges"][i][1]} for i in range(len(input_dict["edges"]))]


## CF
sink_data = np.zeros(365)
tmp = np.array(input_dict["lmax"])
lmax = {}
for i in range(tmp.shape[1]):
    lmax[input_dict["agent_ids"][i]] = tmp[:, i].tolist()

lmax.pop("producer_1")
lmax



strms = [{"sink_id": "aaaa" + str(i), 
            "streams" : [{
                "stream_id": input_dict["agent_ids"][i],
                "hourly_stream_capacity" : lmax[input_dict["agent_ids"][i]]
            } for i in range(len(input_dict["agent_ids"])-1)] }]

strms


## TEO
input_data["teo-module"] = {}

tmp = np.array(input_dict["gmax"])
gmax = {}
for i in range(tmp.shape[1]):
    gmax[input_dict["agent_ids"][i]] = tmp[:, i].tolist()

gmax.pop("consumer_1")
gmax


producers = (input_dict["agent_ids"])[:2] + input_dict["agent_ids"][3:4]

AccNewCap = [{'NAME': 'AccumulatedNewCapacity', 
    'VALUE': mean(gmax[producers[i]]),
    'TECHNOLOGY': producers[i], 
    'YEAR': '2002'} for i in range(len(producers))]

AccNewCap