import pandas as pd

# import own modules
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings, Network
from ...long_term.market_functions.centralized_market import make_centralized_market
from ...long_term.market_functions.decentralized_market import make_decentralized_market


def run_longterm_market(input_dict):
    # create input structures
    settings = MarketSettings(prod_diff=input_dict['prod_diff_option'], market_design=input_dict['md'],
                              horizon_b=input_dict['horizon_basis'],
                              recurr=input_dict['recurrence'], data_prof=input_dict['data_profile'],
                              ydr=input_dict['yearly_demand_rate'])

    agent_data = AgentData(settings=settings, name=input_dict['agent_ids'],
                           #a_type=input_dict['agent_types'],
                            gmax=input_dict['gmax'],
                           lmax=input_dict['lmax'],
                           cost=input_dict['cost'], util=input_dict['util'], co2=input_dict['co2_emissions']
                           )

    gis_data = pd.DataFrame(data=input_dict['gis_data'])
    network = Network(agent_data=agent_data, gis_data=gis_data)

    # construct and solve market -----------------------------
    if settings.market_design == "centralized":
        result = make_centralized_market(agent_data=agent_data, settings=settings)
    elif settings.market_design == "decentralized":
        result = make_decentralized_market(agent_data=agent_data, settings=settings,
                                           network=network)
    else:
        raise ValueError("settings.market_design has to be in [centralized, decentralized]")

    # convert result to right format
    result_dict = result.convert_to_dicts()

    return result_dict
