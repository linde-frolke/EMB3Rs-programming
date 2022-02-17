"""
script that makes input datastructures, then applies market functions
"""
import json
import numpy as np
import pandas as pd
import os
import sys

# import own modules
from market_module.short_term.datastructures.inputstructs import AgentData, MarketSettings, Network
from market_module.short_term.market_functions.pool_market import make_pool_market
from market_module.short_term.market_functions.p2p_market import make_p2p_market

# import own modules
from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market


# TEST POOL #######################################################################################
# setup inputs --------------------------------------------
user_input = {'md': 'pool',
              'offer_type': 'simple',
              'prod_diff': 'noPref', 
              "network": None,
              "el_dependent": False,
              'nr_of_hours': 12,
              "objective": None, 
              "community_settings": None, 
              "block_offer": None,
              "is_in_community" : None, 
              "chp_pars": None, 
              "el_price": None, 
              "start_datetime": "31-01-2002",
              "util": [[40, 42, 35, 25], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0],
                           [34, 44, 55, 0], [29, 33, 45, 0], [40, 55, 33, 0],
                           [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]]
            }

# read file --------------
fn = "/home/linde/Documents/2019PhD/EMB3Rs/module_integration/AccumulatedNewCapacity.json"
f = open(fn, "r")
teo_vals = json.load(f)
# make a dummy 
teo_output = {"AccumulatedNewCapacity": teo_vals, "AnnualVariableOperatingCost": teo_vals, 
                "ProductionByTechnologyAnnual": teo_vals}
AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])
AnnualVariableOperatingCost = pd.json_normalize(teo_output["AnnualVariableOperatingCost"])
ProductionByTechnologyAnnual = pd.json_normalize(teo_output["ProductionByTechnologyAnnual"])

# extract source names:
source_names = AccumulatedNewCapacity.TECHNOLOGY.unique()
source_names = source_names[pd.notna(source_names)]


# make input dict ------------------
# TODO have lmin and gmin set to zero
# TODO remove agent_types from inputs
# 

# extract day month year
day, month, year = [int(x) for x in start_datetime.split("-")]
year = 2000 # must be one of the years that the TEO simulates for
year = AccumulatedNewCapacity.YEAR.min()



input_dict = {
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
                  'util': [[40, 42, 35, 25], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0],
                           [34, 44, 55, 0], [29, 33, 45, 0], [40, 55, 33, 0],
                           [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]],
                  'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                  'is_in_community': 'none',  # allowed values are 'none' or boolean array of size (nr_of_agents)
                  'block_offer': 'none',
                  'is_chp': 'none',  # allowed values are 'none' or a list with ids of agents that are CHPs
                  'chp_pars': 'none',
                  'gis_data': 'none'
                  }

    settings, agent_data, network, result_dict = run_shortterm_market(input_dict=input_dict)

    # MAIN RESULTS

    # Shadow price per hour
    print(result_dict['shadow_price'])

    # Energy dispatch
    print(result_dict['Pn'])

    # Energy dispatch
    print(result_dict['Tnm'])

    # Settlement
    print(result_dict['settlement'])

    # Social welfare
    print(result_dict['social_welfare_h'])

    # Quality of Experience (QoE)
    print(result_dict['QoE'])