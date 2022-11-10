import pandas as pd
from ast import literal_eval

# import own modules
from ...long_term.datastructures.inputstructs import AgentData, MarketSettings
from ...long_term.market_functions.centralized_market import make_centralized_market


from ...cases.exceptions.module_runtime_exception import ModuleRuntimeException
from ...cases.exceptions.module_validation_exception import ModuleValidationException


def run_longterm_market(input_dict):
    # create input structures
    try:
        settings = MarketSettings(product_diff=input_dict['prod_diff_option'], market_design=input_dict['md'],
                                  horizon_basis=input_dict['horizon_basis'],
                                  recurrence=input_dict['recurrence'], data_profile=input_dict['data_profile'],
                                  ydr=input_dict['yearly_demand_rate'], start_datetime=input_dict['start_datetime'],
                                  solver=input_dict['solver'])
    except ModuleValidationException as msg:
        raise print(msg)

    agent_data = AgentData(settings=settings, name=input_dict['agent_ids'],
                           #a_type=input_dict['agent_types'],
                            gmax=input_dict['gmax'],
                           lmax=input_dict['lmax'],
                           cost=input_dict['cost'], util=input_dict['util'], co2=input_dict['co2_emissions']
                           )

    # construct and solve market -----------------------------
    if settings.market_design == "centralized":
        result = make_centralized_market(agent_data=agent_data, settings=settings)
    elif settings.market_design == "decentralized":
        from ...long_term.market_functions.decentralized_market import make_decentralized_market #TODO: #27/07/2022 Moved this, check if is working for the decentralized
        from ...long_term.datastructures.inputstructs import Network #TODO: #27/07/2022 Moved this, check if is working for the decentralized

        gis_data = pd.DataFrame(data=input_dict['gis_data'])
        # convert string to tuple
        gis_data["from_to"] = [literal_eval(x) for x in gis_data["from_to"]]

        network = Network(agent_data=agent_data, gis_data=gis_data)
        result = make_decentralized_market(agent_data=agent_data, settings=settings,
                                           network=network)
    else:
        raise ValueError("settings.market_design has to be in [centralized, decentralized]")

    # convert result to right format
    result_dict = result.convert_to_dicts()

    return result_dict
