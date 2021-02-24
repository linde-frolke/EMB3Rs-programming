"""
script that makes input datastructures, then applies market functions
"""
import numpy as np
import pandas as pd

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.pool_market import make_pool_market
from market_functions.p2p_market import make_p2p_market
from ast import literal_eval

# TEST POOL #######################################################################################
# =============================================================================
# md = "pool"
# =============================================================================
# setup inputs --------------------------------------------
# make settings object
# =============================================================================
# settings = MarketSettings(nr_of_hours=12, offer_type="simple", prod_diff="noPref", market_design=md)
# =============================================================================
# =============================================================================
# settings = MarketSettings(nr_of_hours=12, offer_type="energyBudget", prod_diff="noPref", market_design=md)
# # make agent data object
# =============================================================================
agent_ids = ["sérgio", "linde", "tiago1", "tiago2"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]
# =============================================================================
# =============================================================================
# agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
#                         gmin=np.zeros((settings.nr_of_h, 4)), gmax=np.ones((settings.nr_of_h, 4)),
#                         lmin=np.zeros((settings.nr_of_h, 4)), lmax=np.ones((settings.nr_of_h, 4)),
#                         cost=np.ones((settings.nr_of_h, 4)), util=np.ones((settings.nr_of_h, 4)))
# =============================================================================
# # make network data object
# # first get gis_data and convert from/to to tuple
# =============================================================================
# gis_data = pd.read_csv("test_setup_scripts/Results_GIS.csv")
# gis_data["From/to"] = [literal_eval(i) for i in gis_data["From/to"]]
# network = Network(agent_data=agent_data, gis_data=gis_data)
# network.distance
# network.losses
# =============================================================================
# 
# # set model name
# name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)
# 
# # construct and solve market -----------------------------
# if settings.market_design == "pool":
#     result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
# elif settings.market_design == "community":
#     print("the " + settings.market_design + " market is not implemented yet")
# elif settings.market_design == "p2p":  # P2P should be here
#     result = make_p2p_market(name="test", agent_data=agent_data, settings=settings)
# else:
#     raise ValueError("settings.market_design has to be in [p2p, community, pool")
# 
# # see the result object
# result.name
# result.joint
# result.Ln
# 
# # TEST P2P ########################################################################################
# md = "p2p"
# # setup inputs --------------------------------------------
# settings = MarketSettings(nr_of_hours=12, offer_type="simple", prod_diff="noPref", market_design=md)
# # =============================================================================
# # settings = MarketSettings(nr_of_hours=12, offer_type="energyBudget", prod_diff="noPref", market_design=md)
# # =============================================================================
# 
# # set model name
# name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)
# 
# # construct and solve market -----------------------------
# if settings.market_design == "pool":
#     result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
# elif settings.market_design == "community":
#     print("the " + settings.market_design + " market is not implemented yet")
# elif settings.market_design == "p2p":  # P2P should be here
#     result = make_p2p_market(name="test", agent_data=agent_data, settings=settings)
# else:
#     raise ValueError("settings.market_design has to be in [p2p, community, pool")
# 
# # see the result object
# result.name
# result.joint
# result.Ln
# =============================================================================

# TEST P2P WITH PREFERENCES ########################################################################################
md = "pool"
# setup inputs --------------------------------------------
agent_ids = ["sérgio", "linde", "tiago1", "tiago2"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]

settings = MarketSettings(nr_of_hours=12, offer_type="simple", prod_diff="co2Emissions", market_design=md)
name = "test_" + str(settings.market_design) + "_" + str(settings.offer_type) + "_" + str(settings.product_diff)

# agent data
random_co2 = np.random.rand(len(agent_ids))
random_co2 = random_co2/sum(random_co2) #percentage
random_co2 = np.tile(random_co2,(len(agent_ids),1)) #Dimension [len(agents_ids) x nr_of_h] 


agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=np.zeros((settings.nr_of_h, 4)), gmax=np.ones((settings.nr_of_h, 4)),
                       lmin=np.zeros((settings.nr_of_h, 4)), lmax=np.ones((settings.nr_of_h, 4)),
                       cost=np.ones((settings.nr_of_h, 4)), util=np.ones((settings.nr_of_h, 4)),
                       co2=random_co2)

gis_data = pd.read_csv("test_setup_scripts/Results_GIS.csv")
gis_data["From/to"] = [literal_eval(i) for i in gis_data["From/to"]]
network = Network(agent_data=agent_data, gis_data=gis_data)
network.distance
network.losses






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

# see the result object
result.name
result.joint
result.Ln