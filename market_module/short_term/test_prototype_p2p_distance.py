"""
script that makes input datastructures, then applies market functions
"""
import numpy as np
import pandas as pd
import os
import sys

# make sure it can find modules
sys.path.insert(0, os.getcwd() + '/short_term/')

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.pool_market import make_pool_market
from market_functions.p2p_market import make_p2p_market
from ast import literal_eval


# TEST P2P #######################################################################################
# setup inputs --------------------------------------------
user_input = {'md': 'p2p',
              'nr_of_hours': 12,
              'offer_type': 'simple',
              'prod_diff': 'networkDistance'}
agent_ids = {'agent_ids': ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"]}
agent_types = {'agent_types': ["prosumer", "prosumer", "consumer", "producer"]}

settings = MarketSettings(nr_of_hours=user_input['nr_of_hours'], offer_type=user_input['offer_type'],
                          prod_diff=user_input['prod_diff'], market_design=user_input['md'])
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)

# DATA
block_offer = {'prosumer_1': [[0, 1]], 'producer_1': [[3, 4, 5, 6], [10, 11]]}
co2_emissions = {'co2_emissions': [1, 1.1, 0, 1.8]}
gmin = {'gmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
gmax = {'gmax': [[1, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3], [0, 0, 0, 0], [1, 1, 0, 1], [2, 3, 0, 1], [4, 2, 0, 5],
                 [3, 4, 0, 4], [1, 5, 0, 3],
                 [0, 0, 0, 0], [1, 1, 0, 1], [2, 3, 0, 1]]}
lmin = {'lmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0],
                 [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]}
lmax = {'lmax': [[2, 2, 1, 0], [2, 1, 0, 0], [1, 2, 1, 0], [3, 0, 2, 0], [1, 1, 4, 0], [2, 3, 3, 0], [4, 2, 1, 0],
                 [3, 4, 2, 0], [1, 5, 3, 0],
                 [0, 0, 5, 0], [1, 1, 3, 0], [2, 3, 1, 0]]}
cost = {'cost': [[24, 25, 45, 30], [31, 24, 0, 24], [18, 19, 0, 32], [0, 0, 0, 0], [20, 25, 0, 18], [25, 31, 0, 19],
                 [24, 27, 0, 22], [32, 31, 0, 19],
                 [15, 25, 0, 31], [0, 0, 0, 0], [19, 20, 0, 21], [22, 33, 0, 17]]}
util = {'util': [[40, 42, 35, 25], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0], [34, 44, 55, 0], [29, 33, 45, 0],
                 [40, 55, 33, 0],
                 [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]]}

agent_data = AgentData(settings=settings, name=agent_ids['agent_ids'], a_type=agent_types['agent_types'],
                       gmin=gmin['gmin'], gmax=gmax['gmax'],
                       lmin=lmin['lmin'], lmax=lmax['lmax'],
                       cost=cost['cost'], util=util['util'],
                       co2=co2_emissions['co2_emissions'], block_offer=block_offer)

gis_data = {'From/to': [(0, 1), (1, 2), (1, 3)],
            'Losses total [W]': [22969.228855, 24122.603833, 18138.588662],
            'Length': [1855.232413, 1989.471069, 1446.688900],
            'Total_costs': [1.848387e+06, 1.934302e+06, 1.488082e+06]}
gis_data = pd.DataFrame(data=gis_data)

network = Network(agent_data=agent_data, gis_data=gis_data, settings=settings)

# set model name
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)
# construct and solve market -----------------------------
if settings.market_design == "pool":
    result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    print("the " + settings.market_design + " market is not implemented yet")
elif settings.market_design == "p2p":  # P2P should be here
    result = make_p2p_market(name="test", agent_data=agent_data, settings=settings, network=network)
else:
    raise ValueError("settings.market_design has to be in [p2p, community, pool]")

# MAIN RESULTS

# Shadow price per hour
print(result.shadow_price)

# Energy dispatch
print(result.Pn)

# Settlement
print(result.settlement)

# Social welfare
print(result.social_welfare_h)

# Quality of Experience (QoE)
print(result.QoE)
