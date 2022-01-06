"""
function that makes input structures, runs short term market, makes result object, and returns dictionary outputs
the inputs are also in the dictionary format
"""
import pandas as pd

# import own modules
from ...short_term.datastructures.inputstructs import AgentData, MarketSettings, Network
from ...short_term.market_functions.pool_market import make_pool_market
from ...short_term.market_functions.p2p_market import make_p2p_market
from ...short_term.market_functions.community_market import make_community_market


def run_shortterm_market(input_dict):
    """
    :param input_dict: could be like this:
    
    input_dict = {'sim_name': 'name_of_simulation',
                  'md': 'pool',  # other options are  'p2p' or 'community'
                  'nr_of_hours': 12,
                  'offer_type': 'simple',
                  'prod_diff': 'noPref',
                  'network': 'none',
                  'el_dependent': 'false',
                  'el_price': 'none',
                  'agent_ids': ["prosumer_1",
                            "prosumer_2", "consumer_1", "producer_1"],
                  'agent_types': ["prosumer", "prosumer", "consumer", "producer"],
                  'objective': 'none', # objective for community
                  'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, # or values instead
                  'gmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'gmax': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'lmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'lmax': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'cost': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'util': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], 
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                           [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], 
                  'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                  'is_in_community': 'none',  # allowed values are 'none' or boolean array of size (nr_of_agents) 
                  'block_offer': 'none',  # allowed values are 'none' or list of lists for each agent,
                                e.g {'prosumer_1': [[0, 1]], 'producer_1': [[3, 4, 5, 6], [10, 11]]}
                  'is_chp': 'none', # allowed values are 'none' or a list with ids of agents that are CHPs
                  'chp_pars': 'none',  # a dictionary of dictionaries, including parameters for each agent in is_chp.
                                                        # {'agent_1' : {'rho' : 1.0, 'r' : 0.15, ...},
                                                        #  'agent_2' : {'rho' : 0.8, 'r' : 0.10, ...} }
                  'gis_data': 'none'  # or dictionary of format: 
                            # {'From/to': [(0, 1), (1, 2), (1, 3)],
                            #  'Losses total [W]': [22969.228855, 24122.603833, 18138.588662],
                            #  'Length': [1855.232413, 1989.471069, 1446.688900],
                            #  'Total_costs': [1.848387e+06, 1.934302e+06, 1.488082e+06]}
                  }
    """
    # convert some of the inputs ----------------------------------
    el_dependent = False
    if input_dict["el_dependent"] == "true":
        el_dependent = True

    for str_ in ['network', 'el_price', 'block_offer', 'is_chp', 'chp_pars']:
        if input_dict[str_] == 'none':
            input_dict[str_] = None

    for str_ in ['g_peak', 'g_exp', 'g_imp']:
        if input_dict['community_settings'][str_] == 'none':
            input_dict['community_settings'][str_] = None

    # create Settings object ---------------------------------------
    settings = MarketSettings(nr_of_hours=input_dict['nr_of_hours'], offer_type=input_dict['offer_type'],
                              prod_diff=input_dict['prod_diff'], market_design=input_dict['md'],
                              network_type=input_dict['network'], el_dependent=el_dependent,
                              el_price=input_dict['el_price'])

    if settings.market_design == "community":
        settings.add_community_settings(input_dict['objective'],
                                        g_peak=input_dict['community_settings']['g_peak'],
                                        g_exp=input_dict['community_settings']['g_exp'],
                                        g_imp=input_dict['community_settings']['g_imp'])

    # create AgentData object
    agent_data = AgentData(settings=settings, name=input_dict['agent_ids'], a_type=input_dict['agent_types'],
                           gmin=input_dict['gmin'], gmax=input_dict['gmax'],
                           lmin=input_dict['lmin'], lmax=input_dict['lmax'],
                           cost=input_dict['cost'], util=input_dict['util'],
                           co2=input_dict['co2_emissions'],
                           is_in_community=input_dict['is_in_community'],
                           block_offer=input_dict['block_offer'], is_chp=input_dict['is_chp'],
                           chp_pars=input_dict['chp_pars'], default_alpha=10.0
                           )
    # create Network object
    if input_dict['gis_data'] == "none":
        gis_data = None
    else:
        gis_data = pd.DataFrame(data=input_dict['gis_data'])
    network = Network(agent_data=agent_data, gis_data=gis_data, settings=settings)

    # run market
    # construct and solve market -----------------------------
    if settings.market_design == "pool":
        result = make_pool_market(name="test", agent_data=agent_data, settings=settings, network=network)
    elif settings.market_design == "community":
        result = make_community_market(name="test_comm", agent_data=agent_data, settings=settings)
    elif settings.market_design == "p2p":
        result = make_p2p_market(name="test", agent_data=agent_data, settings=settings, network=network)

    # convert result to dict
    result_dict = result.convert_to_dicts()

    return settings, agent_data, network, result_dict
