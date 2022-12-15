# test community
import numpy as np
import pandas as pd
import os
import sys

# import own modules
from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market
from market_module.short_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs


input_dict = {#'sim_name': 'test_community_autonomy',
                  'md': 'community',  # other options are  'p2p' or 'community'
                  'nr_of_hours': 12,
                  'offer_type': 'simple',
                  'prod_diff': 'noPref',
                  'network': 'none',
                  'el_dependent': 'false',  # can be false or true
                  'el_price': 'none',
                  'agent_ids': ["prosumer_1",
                                "prosumer_2", "consumer_1", "producer_1"],
                  'agent_types': ["prosumer", "prosumer", "consumer", "producer"],
                  'objective': 'autonomy',  # objective for community
                  'community_settings': {'g_peak': 'none', 'g_exp': 1, 'g_imp': 2},
                #   'gmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                #            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                #            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  'gmax': [[1, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3], [0, 0, 0, 0], [1, 1, 0, 1],
                           [2, 3, 0, 1], [4, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3],
                           [0, 0, 0, 0], [1, 1, 0, 1], [2, 3, 0, 1]],
                #   'lmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                #            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                #            [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                  'lmax': [[2, 2, 1, 0], [2, 1, 0, 0], [1, 2, 1, 0], [3, 0, 2, 0], [1, 1, 4, 0],
                           [2, 3, 3, 0], [4, 2, 1, 0], [3, 4, 2, 0], [1, 5, 3, 0], [0, 0, 5, 0],
                           [1, 1, 3, 0], [2, 3, 1, 0]],
                  'cost': [[24, 25, 45, 30], [31, 39, 0, 42], [18, 19, 0, 32], [0, 0, 0, 0],
                           [20, 25, 0, 18], [25, 31, 0, 19], [24, 27, 0, 22], [32, 31, 0, 19],
                           [15, 25, 0, 31], [0, 0, 0, 0], [19, 20, 0, 21], [22, 33, 0, 17]],
                  'util': [[40, 42, 35, 25], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0],
                           [34, 44, 55, 0], [29, 33, 45, 0], [40, 55, 33, 0],
                           [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]],
                  'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                  'is_in_community': [False, False, True, True],# [True, True, False, False], #
                  'block_offer': 'none',
                  'is_chp': 'none',  # allowed values are 'none' or a list with ids of agents that are CHPs
                  'chp_pars': 'none',
                   'gis_data':
                      {'from_to': ['(0, 1)', '(1, 2)', '(1, 3)'],
                       'losses_total': [22969.228855, 24122.603833, 18138.588662],
                       'length': [1855.232413, 1989.471069, 1446.688900],
                       'total_costs': [1.848387e+06, 1.934302e+06, 1.488082e+06]},
                  'nodes' : "none",
                  'edges' : "none"
                  }


result_dict = run_shortterm_market(input_dict=input_dict)

pd.DataFrame(result_dict["settlement"])
pd.DataFrame(result_dict["settlement"]).sum(axis=1)
pd.DataFrame(result_dict["Pn"])
pd.DataFrame(result_dict["Tnm"])
pd.DataFrame(result_dict["shadow_price"])

pd.DataFrame(result_dict["q_imp"])
pd.DataFrame(result_dict["q_exp"])
pd.DataFrame(result_dict["q_comm"])

pd.DataFrame(result_dict["Pn"]).iloc[:, [input_dict["is_in_community"][i] == False for i in range(4)]]


settlement = pd.DataFrame(index=range(12), columns=input_dict["agent_ids"])
for agent in range(4):
  if input_dict['is_in_community'][agent]:
    settlement.iloc[:,agent] = (pd.DataFrame(result_dict["shadow_price"])["community"].values * pd.DataFrame(result_dict["q_comm"]).iloc[:,agent].values + 
            pd.DataFrame(result_dict["shadow_price"])["export"].values * pd.DataFrame(result_dict["q_exp"]).iloc[:,agent].values -
            pd.DataFrame(result_dict["shadow_price"])["import"].values * pd.DataFrame(result_dict["q_imp"]).iloc[:,agent].values  
          )
  else:
    settlement.iloc[:,agent] = pd.DataFrame(result_dict["shadow_price"])["non-community"].values * pd.DataFrame(result_dict["Pn"]).iloc[:,agent].values

(settlement.sum(axis=1) > 0).any()