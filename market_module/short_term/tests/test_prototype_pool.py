"""
test function that makes input datastructures, then applies market functions
for pool market with all settings to default.
"""

# import own modules
from audioop import avg
from lib2to3.pytree import convert
from ...short_term.market_functions.run_shortterm_market import run_shortterm_market
from ...short_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs
import numpy as np
import pandas as pd
import json
import datetime


def test_pool():
    print("running test_pool().............................................")
    # TEST POOL ###############################################################################
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
                  'gis_data': 'none',
                  'nodes' : ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"],
                  'edges' : [("producer_1","consumer_1"), ("producer_1","prosumer_1"),
                             ("prosumer_1","prosumer_2"), ]
                  }

    result_dict = run_shortterm_market(input_dict=input_dict)

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

    # test using "convert_user_and_module_inputs()" -------------------------------------------------
    # convert some inputs to platform format
    tmp = np.array(input_dict["util"])
    util = {}
    for i in range(tmp.shape[1]):
        util[input_dict["agent_ids"][i]] = tmp[:, i].tolist()
    util.pop("producer_1")
    util.pop("prosumer_1") # make prosumer_1 a producer
    
    # initialize input data dictionary
    input_data = {}
    # add platform data 
    input_data["platform"] = {
        "md": "pool",
        "offer_type": "simple",
        "prod_diff": "noPref", 
        "network" : "none",
        "el_dependent": "false",
        "nr_of_hours": 12,
        "objective": "none",
        "community_settings": {
            "g_peak": "none", 
            "g_exp": "none",
            "g_imp": "none"
            },
        "block_offer": "none",
        "is_in_community": "none",
        "chp_pars": "none",
        "el_price": "none",
        "start_datetime": "31-01-2002", 
        "util": util
        }
    # extract day month year
    day, month, year = [int(x) for x in input_data["platform"]["start_datetime"].split("-")]
    as_date = datetime.datetime(year=year, month=month, day=day)
    start_hourofyear = as_date.timetuple().tm_yday * 24   # start index if selecting from entire year of hourly data.
    end_hourofyear = start_hourofyear + input_data["platform"]["nr_of_hours"] # end index if selecting from entire year of hourly data.
    
    # add gis data  (load format from file...) -------------------------------
    res_sources_sinks = "none" # pd.DataFrame(input_dict["gis_data"]).to_dict("records")
    netsolnodes = [{"osmid" : input_dict["nodes"][i]} for i in range(len(input_dict["nodes"]))]
    netsoledge = [{"from" : input_dict["edges"][i][0], 
                   "to" : input_dict["edges"][i][1]} for i in range(len(input_dict["edges"]))]

    input_data["gis-module"] = {"res_sources_sinks" : res_sources_sinks,
                                "network_solution_nodes" : netsolnodes,
                                "network_solution_edges" : netsoledge
                                }

    # add CF data ---
    tmp = np.array(input_dict["lmax"])
    lmax = {}
    for i in range(tmp.shape[1]):
        lmax_sel = tmp[:, i].tolist()
        tmp2 = [0] * 365 * 24
        # print(lmax_sel)
        tmp2[start_hourofyear:end_hourofyear] = lmax_sel
        # print(tmp2)
        lmax[input_dict["agent_ids"][i]] = tmp2
        

    lmax.pop("producer_1")
    lmax.pop("prosumer_1")
    strms = [{"sink_id": "aaaa" + str(i), 
            "streams" : [{
                "stream_id": input_dict["agent_ids"][i],
                "hourly_stream_capacity" : lmax[input_dict["agent_ids"][i]]
            }  for i in [1,2]] }]
    input_data["cf-module"] = {"all_sinks_info": {"sinks": strms}}

    # add TEO data ---
    producers = (input_dict["agent_ids"])[:1] + input_dict["agent_ids"][3:4]
    # GMAX
    tmp = np.array(input_dict["gmax"])
    gmax = {}
    for i in range(tmp.shape[1]):
        gmax[input_dict["agent_ids"][i]] = tmp[:, i].tolist()
    gmax.pop("consumer_1")
    gmax.pop("prosumer_2")
    
    AccNewCap = [{'NAME': 'AccumulatedNewCapacity', 
        'VALUE': max(gmax[producers[i]]),
        'TECHNOLOGY': producers[i], 
        'YEAR': '2002'} for i in range(len(producers))]
    # COST
    tmp = np.array(input_dict["cost"])
    cost = {}
    for i in range(tmp.shape[1]):
        cost[input_dict["agent_ids"][i]] = tmp[:, i].tolist()
    cost.pop("consumer_1")
    cost.pop("prosumer_2")
    AnnVarOp = [{'NAME': 'AnnualVariableOperatingCost', 
        'VALUE': sum(cost[producers[i]]) / len(cost[producers[i]]),
        'TECHNOLOGY': producers[i], 
        'YEAR': '2002'} for i in range(len(producers))]
    # Total prod
    ProdTechAn = [{'NAME': 'ProductionByTechnologyAnnual', 
        'VALUE': 1,
        'TECHNOLOGY': producers[i], 
        'YEAR': '2002'} for i in range(len(producers))]
    
    # CO2 
    AnTechEm = [{'NAME': 'AnnualTechnologyEmission', 
        'VALUE': 0,
        'EMISSION': "CO2",
        'TECHNOLOGY': producers[i], 
        'YEAR': '2002'} for i in range(len(producers))]


    input_data["teo-module"] = {"AccumulatedNewCapacity": AccNewCap,
                                "AnnualVariableOperatingCost": AnnVarOp, 
                                "ProductionByTechnologyAnnual": ProdTechAn,
                                "AnnualTechnologyEmission": AnTechEm
                                }


    # convert inputs to wanted scheme -------------------------------------------
    inputs_converted = convert_user_and_module_inputs(input_data=input_data)
    # run the market ------------------------------------------------------------
    result_dict2 = run_shortterm_market(input_dict=inputs_converted)

    # MAIN RESULTS

    # Shadow price per hour
    print(result_dict2['shadow_price'])

    # Energy dispatch
    print(result_dict2['Pn'])

    # Energy dispatch
    print(result_dict2['Tnm'])

    # Settlement
    print(result_dict2['settlement'])

    # Social welfare
    print(result_dict2['social_welfare_h'])

    # Quality of Experience (QoE)
    print(result_dict2['QoE'])


    print("finished test_pool().............................................")
