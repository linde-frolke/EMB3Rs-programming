"""
script that makes input datastructures, then applies market functions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import tkinter
# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.pool_market import make_pool_market
from market_functions.p2p_market import make_p2p_market
from market_functions.community_market import make_community_market
from ast import literal_eval
mpl.use('TkAgg')

# TEST POOL #######################################################################################
md = "pool"
# setup inputs --------------------------------------------
# make settings object
settings = MarketSettings(nr_of_hours=1, offer_type="simple", prod_diff="noPref", market_design=md)
#settings = MarketSettings(nr_of_hours=12, offer_type="energyBudget", prod_diff="noPref", market_design=md)
# make agent data object
agent_ids = ["sérgio", "linde", "tiago1", "tiago2"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]

lmin = np.array([1.0, 2.0, 3.0, 0.0])
lmax = np.array([7.0, 10.0, 15.0, 0.0])
gmin = np.array([0.0, 5.0, 0.0, 10.0])
gmax = np.array([3.0, 7.0, 0.0, 15.0])
cost = np.array([1.0, 0.5, 0.0, 4.0])
util = np.array([3.0, 4.0, 6.0, 0.0])

agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=np.zeros((settings.nr_of_h, 4)), gmax=gmax,
                       lmin=lmin, lmax=lmax, cost=cost, util=util)
# set model name
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)

# construct and solve market -----------------------------
if settings.market_design == "pool":
    result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    print("the " + settings.market_design + " market is not implemented yet")
elif settings.market_design == "p2p":  # P2P should be here
    result = make_p2p_market(name="test", agent_data=agent_data, settings=settings)

# see the result object
result.name
result.joint
result.Ln
result.shadow_price

output_plot_file = "./figures/test.png"
result.plot_market_clearing(period=0, agent_data=agent_data, outfile=output_plot_file)
