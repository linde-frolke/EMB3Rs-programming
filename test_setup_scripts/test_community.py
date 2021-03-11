import numpy as np
import pandas as pd

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.pool_market import make_pool_market
from market_functions.p2p_market import make_p2p_market
from market_functions.community_market import make_community_market
from ast import literal_eval
import os
os.getcwd()
#os.chdir("pickled_data")

# TEST COMMUNITY ##################################################################################################
md = "community"
# setup inputs --------------------------------------------
# make settings object
settings = MarketSettings(nr_of_hours=12, offer_type="simple", prod_diff="noPref", market_design=md)
settings.add_community_settings(objective="autonomy")
# make agent data object
agent_ids = ["sérgio", "linde", "tiago1", "tiago2"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]
agent_community = [False, False, True, True]
lmin = np.zeros((settings.nr_of_h, 4))
lmin[:, 2] = 0
gmax = np.ones((settings.nr_of_h, 4))
gmax[:, 2] = 0

agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=np.zeros((settings.nr_of_h, 4)), gmax=gmax,
                       lmin=lmin, lmax=np.ones((settings.nr_of_h, 4)),
                       cost=np.ones((settings.nr_of_h, 4)), util=np.ones((settings.nr_of_h, 4)),
                       is_in_community=agent_community)
agent_data.agent_is_in_community
agent_data.C
agent_data.notC
settings.community_objective

# set model name
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.community_objective)

# construct and solve market -----------------------------
if settings.market_design == "pool":
    result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    result = make_community_market(name="test_comm", agent_data=agent_data, settings=settings)
elif settings.market_design == "p2p":  # P2P should be here
    result = make_p2p_market(name="test", agent_data=agent_data, settings=settings)

# see the result object
result.name
result.Ln
result.Gn
result.Tnm

result.shadow_price
result.settlement

result.save_as_pickle()


# test the other community setup
settings.add_community_settings(objective="peakShaving", g_peak=10.0**2)
# set model name
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.community_objective)

# construct and solve market -----------------------------
if settings.market_design == "pool":
    result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    result = make_community_market(name="test_comm", agent_data=agent_data, settings=settings)
elif settings.market_design == "p2p":  # P2P should be here
    result = make_p2p_market(name="test", agent_data=agent_data, settings=settings)

# see the result object
result.name
result.Ln
result.Gn


result.shadow_price