"""
script that makes input datastructures, then tests p2p market
"""
import numpy as np
import pandas as pd

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.pool_market import make_pool_market
from market_functions.p2p_market import make_p2p_market
from market_functions.community_market import make_community_market
from ast import literal_eval

## define market settings
md = "p2p"
# setup inputs --------------------------------------------
settings = MarketSettings(nr_of_hours=1, offer_type="simple", prod_diff="noPref", market_design=md)
#settings = MarketSettings(nr_of_hours=12, offer_type="energyBudget", prod_diff="noPref", market_design=md)
# set model name
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)
# make agent data object
agent_ids = ["sérgio", "linde"]  # "tiago1", "tiago2"]
agent_types = ["consumer", "producer"]
lmin = np.array([2.0, 0.0])
lmax = np.array([3.0, 0])
gmin = np.array([0, 1.0])
gmax = np.array([0, 5.0])
cost = np.array([0, 11.0])
util = np.array([15.0, 0])
agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=gmin, gmax=gmax, lmin=lmin, lmax=lmax,
                       cost=cost, util=util)

# construct and solve market -----------------------------
if settings.market_design == "pool":
    result = make_pool_market(name=name, agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    print("the " + settings.market_design + " market is not implemented yet")
elif settings.market_design == "p2p":  # P2P should be here
    result = make_p2p_market(name=name, agent_data=agent_data, settings=settings, network=Network)

# see the result object
result.joint
result.Ln
result.Gn
result.Pn

# result.varnames
# result of trades at time 0
result.Tnm[0]
result.shadow_price[0]

result.