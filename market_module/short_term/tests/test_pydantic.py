## Test pydantic 

from market_module.short_term.datastructures.inputstructs import MarketSettings, AgentData, Network

# good settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", market_design="pool", network_type=None,
                    el_dependent=False, el_price=None)

# bad settings
settings = MarketSettings(nr_of_h="2", offer_type="simple", product_diff="noPref", market_design="pool", 
                    network_type=None,
                    el_dependent=False, el_price=None)

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff=4, market_design=5, network_type=None,
                    el_price=None, el_dependent=True)

# bad settings
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="co2Emissions", 
                market_design="pool", network_type=None,
                el_dependent=False, el_price=None)

# bad community settings 
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=10,
                                gamma_imp=None, 
                                gamma_exp=None)

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=None,
                                gamma_imp=None, 
                                gamma_exp=None)

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="autonomy", 
                                gamma_peak=5,
                                gamma_imp=None, 
                                gamma_exp=-10)
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
                market_design="community", network_type=None,
                el_dependent=False, el_price=None, community_objective="peakShaving", 
                                gamma_peak=None,
                                gamma_imp=5, 
                                gamma_exp=-6)
## Agentdata
# good settings
from market_module.short_term.datastructures.inputstructs import MarketSettings, AgentData, Network

settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", market_design="pool", network_type=None,
                    el_dependent=False, el_price=None)

# good agent data
agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[1,2,3], [1,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)

network = Network(agent_data=agent_data, settings=settings, gis_data=None, 
                      N=None, P=None)

# bad agentdata: missing co2. 
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="co2Emissions", market_design="p2p", network_type=None,
                    el_dependent=False, el_price=None)

agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[1,2,3], [1,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)
# bad agentdata: missing co2. 
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="co2Emissions", market_design="p2p", network_type=None,
                    el_dependent=False, el_price=None)

agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[1,2,3], [1,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)

############# network 
from market_module.short_term.datastructures.inputstructs import MarketSettings, AgentData, Network
import pandas as pd
import numpy as np
from ast import literal_eval

# network good
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
               market_design="pool", 
               network_type=None, el_dependent=False, el_price=None)
agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[6,2,3], [5,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)
gis_data = {'from_to': ['("1", "3")', '("1", "6")'],
            'losses_total': [22969.228855, 24122.603833],
            'length': [1855.232413, 1989.471069],
            'total_costs': [1.848387e+06, 1.934302e+06]}
gis_data = pd.DataFrame(data=gis_data)
# convert string to tuple
gis_data["from_to"] = [literal_eval(x) for x in gis_data["from_to"]]
network = Network(agent_data=agent_data, settings=settings, gis_data=gis_data, 
                      N=None, P=None)
## good network: gis if p2p and preferences chosen
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="losses", 
               market_design="p2p", 
               network_type=None, el_dependent=False, el_price=None)
agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[6,2,3], [5,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)
gis_data = {'from_to': ['("1", "3")', '("1", "6")'],
            'losses_total': [22969.228855, 24122.603833],
            'length': [1855.232413, 1989.471069],
            'total_costs': [1.848387e+06, 1.934302e+06]}
gis_data = pd.DataFrame(data=gis_data)
# convert string to tuple
gis_data["from_to"] = [literal_eval(x) for x in gis_data["from_to"]]
network = Network(agent_data=agent_data, settings=settings, gis_data=gis_data, 
                      N=None, P=None)

## network bad! 
network = Network(agent_data=agent_data, settings=settings, gis_data=None, 
                      N=None, P=None)
## another network bad: no nodes and pipes given 
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
               market_design="pool", network_type="direction", el_dependent=False, el_price=None)
agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[6,2,3], [5,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)
network = Network(agent_data=agent_data, settings=settings, gis_data=None, 
                      N=None, P=None)

# bad gis_data column names and/or agent IDs
gis_data = {'from_to': ['("1", "5")', '("1", "6")'],
            'losses_total': [22969.228855, 24122.603833],
            'lengthG': [1855.232413, 1989.471069],
            'total_costs': [1.848387e+06, 1.934302e+06]}
gis_data = pd.DataFrame(data=gis_data)
# convert string to tuple
gis_data["from_to"] = [literal_eval(x) for x in gis_data["from_to"]]
settings = MarketSettings(nr_of_h=2, offer_type="simple", product_diff="noPref", 
               market_design="p2p", network_type=None, el_dependent=False, el_price=None)
agent_data = AgentData(settings=settings,
                        agent_name=["1", "3", "6"],
                           gmax=[[6,2,3], [5,2,3]],
                           lmax=[[1,2,3],[1,2,3]],
                           cost=[[1,2,3],[1,2,3]], util=[[1,2,3],[1,2,3]],
                           co2=None,
                           is_in_community=None,
                           block=None, is_chp=None,
                           chp_pars=None)
network = Network(agent_data=agent_data, settings=settings, gis_data=gis_data, 
                      N=None, P=None)