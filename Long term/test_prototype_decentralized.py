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

# TEST DECENTRALIZED #######################################################################################
# setup inputs --------------------------------------------
user_input={'md': 'decentralized',
            'horizon_basis': 'months',
            'data_profile': 'daily',
            'recurrence': 2,
            'yearly_demand_rate': 0.05,
            'prod_diff_option':'co2Emissions'
            }
agent_ids = {'agent_ids': ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"]}
agent_types = {'agent_types': ["prosumer", "prosumer", "consumer", "producer"]}  

settings = MarketSettings(prod_diff=user_input['prod_diff_option'], market_design=user_input['md'],
                          horizon_b=user_input['horizon_basis'], recurr=user_input['recurrence'],
                          data_prof=user_input['data_profile'],
                          ydr=user_input['yearly_demand_rate'])

name = "test_" + str(settings.market_design) + "_" + str(settings.product_diff)

#DATA
co2_emissions = {'co2_emissions': np.array([1, 1.1, 0, 1.8])}
gmin = {'gmin':np.zeros([60,4])}
gmax = {'gmax':np.random.uniform(low=1, high=8, size=(60,4))}
lmin = {'lmin':np.zeros([60,4])}
lmax = {'lmax':np.random.uniform(low=1, high=3, size=(60,4))}
 
cost = {'cost':np.random.uniform(low=20, high=30, size=(60,4))}
util = {'util':np.random.uniform(low=25, high=35, size=(60,4))}


agent_data = AgentData(settings=settings, name=agent_ids['agent_ids'], a_type=agent_types['agent_types'],
                       gmin=gmin['gmin'], gmax=gmax['gmax'],
                       lmin=lmin['lmin'], lmax=lmax['lmax'],
                       cost=cost['cost'], util=util['util'], co2=co2_emissions['co2_emissions']
                       )

gis_data = {'From/to':[(0,1),(1,2),(1,3)], 
            'Losses total [W]':[22969.228855, 24122.603833,18138.588662],
            'Length':[1855.232413,1989.471069,1446.688900], 
            'Total_costs':[1.848387e+06,1.934302e+06, 1.488082e+06]}
gis_data = pd.DataFrame(data=gis_data)
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
#print(result.find_best_price(15, 'prosumer_1', agent_data, settings)) #user must select hour and agent_id