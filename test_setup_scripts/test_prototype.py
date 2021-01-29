"""
script that makes input datastructures, then applies market functions
"""
import numpy as np
# import pandas as pd

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings
# from datastructures.resultobject import ResultData
from market_functions.pool_market import make_pool_market

# setup inputs --------------------------------------------
# make settings object
settings = MarketSettings(nr_of_hours=12, offer_type="simple", prod_diff="noPref", market_design="pool")

# make agent data object
agent_ids = ["sérgio", "linde", "tiago1", "tiago2"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]
agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=np.zeros((settings.nr_of_h, 4)), gmax=np.ones((settings.nr_of_h, 4)),
                       lmin=np.zeros((settings.nr_of_h, 4)), lmax=np.ones((settings.nr_of_h, 4)),
                       cost=np.ones((settings.nr_of_h, 4)), util=np.ones((settings.nr_of_h, 4)))

# set model name
name = "test"

# construct and solve market
if settings.market_design == "pool":
    result = make_pool_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "community":
    print("the " + settings.market_design + " market is not implemented yet")
elif settings.market_design == "p2p":  # P2P should be here
    print("the " + settings.market_design + " market is not implemented yet")
else:
    raise ValueError("settings.market_design has to be in [p2p, community, pool")

# see the result object
result.name
result.joint
result.Ln