"""
script that makes input datastructures, then applies market functions
"""
import numpy as np
import pandas as pd

# import own modules
from datastructures.inputstructs import AgentData, MarketSettings, Network
from market_functions.centralized_market import make_centralized_market
from market_functions.decentralized_market import make_decentralized_market
from ast import literal_eval

# TEST CENTRALIZED #######################################################################################
md = "centralized"
# setup inputs --------------------------------------------
agent_ids = ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"]
agent_types = ["prosumer", "prosumer", "consumer", "producer"]
horizon_basis = "months"
recurrence = 2
yearly_demand_rate = 0.05
data_profile = 'daily'

settings = MarketSettings(prod_diff="co2Emissions", market_design=md,
                          horizon_b=horizon_basis, recurr=recurrence, data_prof=data_profile,
                          ydr=yearly_demand_rate)

name = "test_" + str(settings.market_design) + "_" + str(settings.product_diff)


#DATA
co2_emissions = np.array([1, 1.1, 0, 1.8])
gmin = np.zeros([60,4])
gmax = np.random.uniform(low=1, high=8, size=(60,4))
lmin = np.zeros([60,4])
lmax = np.random.uniform(low=1, high=3, size=(60,4))
 
cost = np.random.uniform(low=20, high=30, size=(60,4))
util = np.random.uniform(low=25, high=35, size=(60,4))

# =============================================================================
# gmin=np.zeros((settings.nr_of_h, len(agent_ids)))
# gmax=np.array([[1,2,0,5],[3,4,0,4],[1,5,0,3],[0,0,0,0],[1,1,0,1],[2,3,0,1],[4,2,0,5],[3,4,0,4],[1,5,0,3],
#               [0,0,0,0],[1,1,0,1],[2,3,0,1]])
# lmin=np.zeros((settings.nr_of_h, len(agent_ids)))
# lmax=np.array([[2,2,1,0],[2,1,0,0],[1,2,1,0],[3,0,2,0],[1,1,4,0],[2,3,3,0],[4,2,1,0],[3,4,2,0],[1,5,3,0],
#               [0,0,5,0],[1,1,3,0],[2,3,1,0]])
# cost=np.array([[24,25,45,30],[31,24,0,24],[18,19,0,32],[0,0,0,0],[20,25,0,18],[25,31,0,19],[24,27,0,22],[32,31,0,19],
#                [15,25,0,31],[0,0,0,0],[19,20,0,21],[22,33,0,17]])
# util=np.array([[40,42,35,25],[45,50,40,0],[55,36,45,0],[44,34,43,0],[34,44,55,0],[29,33,45,0],[40,55,33,0],
#                [33,42,38,0],[24,55,35,0],[25,35,51,0],[19,43,45,0],[34,55,19,0]])
# =============================================================================


agent_data = AgentData(settings=settings, name=agent_ids, a_type=agent_types,
                       gmin=gmin, gmax=gmax,
                       lmin=lmin, lmax=lmax,
                       cost=cost, util=util, co2=co2_emissions
                       )

gis_data = pd.read_csv("test_setup_scripts/Results_GIS.csv")
gis_data["From/to"] = [literal_eval(i) for i in gis_data["From/to"]]
network = Network(agent_data=agent_data, gis_data=gis_data)

# set model name
name = "test_" + str(settings.market_design) + "_"  + "_" + str(settings.product_diff)
# construct and solve market -----------------------------
if settings.market_design == "centralized":
    result = make_centralized_market(name="test", agent_data=agent_data, settings=settings)
elif settings.market_design == "decentralized":  
    result = make_decentralized_market(name="test", agent_data=agent_data, settings=settings, network=network)
else:
    raise ValueError("settings.market_design has to be in [centralized or decentralized]")

#MAIN RESULTS
#ADG
print(result.ADG)
print(result.SPM)

#Shadow price per hour
print(result.shadow_price)


#Energy dispatch
#print(result.Pn)
#print(result.Ln)
print(result.Gn)


#Settlement
#print(result.settlement)

#Social welfare
#print(result.social_welfare_h)


#Find the best price
print(result.find_best_price(15, 'prosumer_1', agent_data, settings)) #user must select hour and agent_id