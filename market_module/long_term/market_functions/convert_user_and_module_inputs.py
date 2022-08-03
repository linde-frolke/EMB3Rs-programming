"""
function that converts TEO and CF inputs to the "input_dict" we were expecting
"""
import json
import numpy as np
import pandas as pd
import datetime
import xlrd
from datetime import datetime
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse



def convert_user_and_module_inputs(input_data):
    # user_inputs
    user_input = input_data ### separate dictionary inside

    # Date related
    datetime_date = user_input['user']['start_datetime']
    start_date = parse(datetime_date.strftime('%d-%m-%Y'))
    start_date_str = start_date.strftime('%d-%m-%Y')

    if user_input['user']['horizon_basis'] == 'weeks':
        end_date = start_date + relativedelta(weeks=user_input['user']["recurrence"])
    if user_input['user']['horizon_basis'] == 'months':
        end_date = start_date + relativedelta(months=user_input['user']["recurrence"])
    if user_input['user']['horizon_basis'] == 'years':
        #always sending just one year of hourly data
        end_date = start_date + relativedelta(years=1)

    if user_input['user']['data_profile'] == 'hourly':
        diff = end_date - start_date  # difference
        diff = int(diff.total_seconds()/3600) #difference in hours

    if user_input['user']['data_profile'] == 'daily':
        diff = end_date - start_date  # difference
        # always sending just one year of hourly data
        diff = int(diff.total_seconds()/3600) #difference in hours

    nr_of_hours = diff

#TODO: Check later if this is required for the longterm


    # # get GIS data ----------
    #TODO: do this if we use decentralized market

    # get CF data
    all_sinks_info = input_data["cf-module"]["all_sinks_info"]["sinks"]

    # convert CF inputs -----------------------------------------------------------------------------
    # get some sink info
    nr_of_sinks = len(all_sinks_info)

    sink_ids = []
    map_sink_streams = []

    for i in range(nr_of_sinks):
        sink_ids += [all_sinks_info[i]["sink_id"]]
        map_sink_streams += [[streams["demand_fuel"] for streams in all_sinks_info[i]["streams"]]]

    # get some stream x
    all_stream_ids = [y for x in map_sink_streams for y in x]

    # prep inputs
    lmax_sinks = np.zeros((diff, len(all_stream_ids)))
    for t in range(0, diff):
        count=0
        for sink in range(0, len(all_sinks_info)):
            for stream in range(0,len(all_sinks_info[sink]['streams'])):
                lmax_sinks[t,count] = all_sinks_info[sink]['streams'][stream]['hourly_stream_capacity'][t]
                count+=1


    utility_list = user_input['user']['util']
    util_sinks_t0 = np.array(utility_list)
    util_sinks = np.tile(util_sinks_t0,(diff,1))

    if len(util_sinks_t0) != len(all_stream_ids):
        raise('Utility does not match the sinks size')

    # get TEO data
    teo_output = input_data["teo-module"]

    #Convert TEO inputs
    #TODO: get co2 emissions if we use product differentiation
    ProductionByTechnologyAnnual = pd.DataFrame(teo_output["ProductionByTechnologyMM"])
    VariableOMCost = pd.DataFrame(teo_output["VariableOMCost"])

    #Converting timeslice from str to int
    ProductionByTechnologyAnnual['TIMESLICE'] = ProductionByTechnologyAnnual['TIMESLICE'].astype('int')
    #Getting names
    agent_names = ProductionByTechnologyAnnual.TECHNOLOGY.unique()
    #Getting source names by removing sink names
    source_names = []
    for word in agent_names:
        if 'sink' not in word:
            source_names.append(word)

    nr_of_sources = len(source_names)

    dummy = {'TIMESLICE': range(1, nr_of_hours+1)}
    #Building gmax_sources
    gmax_sources = pd.DataFrame(dummy)
    gmax_sources.set_index('TIMESLICE', inplace=True)

    for source in source_names:
        new_df = ProductionByTechnologyAnnual[(ProductionByTechnologyAnnual['TECHNOLOGY'] == source)]
        new_df['TIMESLICE'] = new_df['TIMESLICE'].apply(int)
        new_df = new_df[['TIMESLICE',"VALUE"]]

        new_df.set_index('TIMESLICE',inplace=True)
        new_df.sort_values(by=['TIMESLICE'], ascending=True, inplace=True)
        new_df.rename(columns={'VALUE': source}, inplace=True)

        gmax_sources = pd.concat([gmax_sources, new_df], axis=1)

    gmax_sources = gmax_sources.fillna(0)
    gmax_sources = gmax_sources[0:nr_of_hours].to_numpy()

    # Building cost_sources
    cost_sources = pd.DataFrame(dummy)
    cost_sources.set_index('TIMESLICE', inplace=True)
    for source in source_names:
        cost_sources[source] = VariableOMCost.loc[(VariableOMCost['TECHNOLOGY'] == source)]["VALUE"].values[0]

    cost_sources = cost_sources.to_numpy()

    # prep inputs
    lmax_sources = np.zeros((nr_of_hours, nr_of_sources))
    util_sources = np.zeros((nr_of_hours, nr_of_sources))
    gmax_sinks = np.zeros((nr_of_hours, len(all_stream_ids)))
    cost_sinks = np.zeros((nr_of_hours, len(all_stream_ids)))

    ## combine in input_dict as we used to have it
    # make input dict ------------------------
    # combine source and sink inputs
    agent_ids = source_names + all_stream_ids
    gmax = np.concatenate((gmax_sources, gmax_sinks), axis=1).tolist()
    lmax = np.concatenate((lmax_sources, lmax_sinks), axis=1).tolist()
    cost = np.concatenate((cost_sources, cost_sinks), axis=1).tolist()
    util = np.concatenate((util_sources, util_sinks), axis=1).tolist()

    # construct input_dict
    input_dict = {
                    'md': user_input['user']['md'],
                    'horizon_basis': user_input['user']["horizon_basis"],
                    'data_profile': user_input['user']["data_profile"],
                    'recurrence': user_input['user']["recurrence"],
                    'yearly_demand_rate': user_input['user']["yearly_demand_rate"],
                    'prod_diff_option': user_input['user']["prod_diff_option"],
                    'agent_ids': agent_ids,
                    'gmax': gmax,
                    'lmax': lmax,
                    'cost': cost,
                    'util': util,
                    'start_datetime': start_date_str,
                    'co2_emissions': None,  # allowed values are 'none' or array of size (nr_of_agents)
                    'gis_data': None,
                    'nodes': None,
                    'edges': None
                    }

    return input_dict
