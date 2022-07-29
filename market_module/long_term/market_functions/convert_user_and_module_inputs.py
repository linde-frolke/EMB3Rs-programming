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
    # dummy = False 

    # user_inputs
    user_input = input_data ### separate dictionary inside

    # Date related
    datetime_date = xlrd.xldate_as_datetime(user_input['user']['start_datetime'], 0)
    #date_format = '%d-%m-%Y'
    #start_date1 =datetime_date.strftime('%d-%m-%Y')
    start_date = parse(datetime_date.strftime('%d-%m-%Y'))
    start_date_str = start_date.strftime('%d-%m-%Y')

    if user_input['user']['horizon_basis'] == 'weeks':
        end_date = start_date + relativedelta(weeks=user_input['user']["recurrence"])
    if user_input['user']['horizon_basis'] == 'months':
        end_date = start_date + relativedelta(months=user_input['user']["recurrence"])
    if user_input['user']['horizon_basis'] == 'years':
        end_date = start_date + relativedelta(years=user_input['user']["recurrence"])


    if user_input['user']['data_profile'] == 'hourly':
        diff = end_date - start_date  # difference
        diff = int(diff.total_seconds()/3600) #difference in hours

    if user_input['user']['data_profile'] == 'daily':
        diff = end_date - start_date  # difference
        diff = int(diff.total_seconds()/3600/24) #difference in days

    nr_of_hours = diff

#TODO: Check later if this is required for the longterm
    # # # extract day month year------------------
    # day, month, year = int(start_date.strftime('%d')), int(start_date.strftime('%m')), int(start_date.strftime('%Y'))
    # as_date = datetime(year=year, month=month, day=day)
    # start_hourofyear = as_date.timetuple().tm_yday * 24   # start index if selecting from entire year of hourly data.
    # end_hourofyear = start_hourofyear + nr_of_hours # end index if selecting from entire year of hourly data.

    # # get GIS data ----------
    # gis_output = input_data["gis-module"]
    # nodes = [x["osmid"] for x in  gis_output["network_solution_nodes"]]
    # edges = pd.DataFrame(gis_output["network_solution_edges"])
    # # gis_output["selected_agents"]  # TODO see if I can use this for something

    # get CF data
    all_sinks_info = input_data["cf-module"]["all_sinks_info"]["sinks"]

    # convert CF inputs -----------------------------------------------------------------------------
    # get some sink info
    nr_of_sinks = len(all_sinks_info)

    #for sink in range(0,nr_of_sinks):
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

    utility = user_input['user']['util'][0].split(",")
    utility_list = []
    for value in utility:
        utility_list.append(int(value))
    util_sinks_t0 = np.array(utility_list)

    util_sinks = np.tile(util_sinks_t0,(diff,1))

    if len(util_sinks_t0) != len(all_stream_ids):
        raise('Utility does not match the sinks size')


    gmax_sinks = np.zeros(np.shape(lmax_sinks))
    cost_sinks = np.zeros(np.shape(lmax_sinks))
    co2_em_sinks = np.zeros(nr_of_sinks) #TODO: check if needed

    # get TEO data
    teo_output = input_data["teo-module"]

#Convert TEO inputs
    #AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])
    ProductionByTechnologyAnnual = pd.json_normalize(teo_output["ProductionByTechnology"])
    VariableOMCost = pd.json_normalize(teo_output["VariableOMCost"])

    #Converting timeslice from str to int
    ProductionByTechnologyAnnual['TIMESLICE'] = ProductionByTechnologyAnnual['TIMESLICE'].astype('int')

    agent_names = ProductionByTechnologyAnnual.TECHNOLOGY.unique()
    #Getting source names by removing sink names
    source_names = []
    for word in agent_names:
        if 'sink' not in word:
            source_names.append(word)

    nr_of_sources = len(source_names)

    gmax_sources = np.zeros((nr_of_hours, nr_of_sources))
    for t in range(0,diff):
        print(t)
        for source in source_names:
            try: #When VALUE=0 it is not providing any value at all
                gmax_sources[t,source_names.index(source)] = (ProductionByTechnologyAnnual.loc[(ProductionByTechnologyAnnual['TIMESLICE'] == t+1)
                                                                                           & (ProductionByTechnologyAnnual['TECHNOLOGY'] == source)].VALUE)
            except:
                pass

    cost_sources = np.zeros((nr_of_hours, nr_of_sources))
    for t in range(0, diff): #only one value is provided
        print(t)
        for source in source_names:
            cost_sources[t, source_names.index(source)] = VariableOMCost.loc[(VariableOMCost['TECHNOLOGY'] == source )].VALUE


    # convert TEO inputs ----------------------------------------------------------------------------
    #AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])
    #AccumulatedNewCapacity["YEAR"] = AccumulatedNewCapacity["YEAR"].astype(int)
    #AnnualVariableOperatingCost = pd.json_normalize(teo_output["AnnualVariableOperatingCost"])
    #AnnualVariableOperatingCost["YEAR"] = AnnualVariableOperatingCost["YEAR"].astype(int)
    #ProductionByTechnologyAnnual = pd.json_normalize(teo_output["ProductionByTechnologyAnnual"])
    #ProductionByTechnologyAnnual["YEAR"] = ProductionByTechnologyAnnual["YEAR"].astype(int)
    #AnnualTechnologyEmission = pd.json_normalize(teo_output["AnnualTechnologyEmission"])
    #AnnualTechnologyEmission["YEAR"] = AnnualTechnologyEmission["YEAR"].astype(int)

    # year must be one of the years that the TEO simulates for
    #if not year in AccumulatedNewCapacity.YEAR.to_list():
        #ValueError("User chosen 'year' must be one of the YEARs provided by TEO")
    
    # extract source names 
    #source_names = AccumulatedNewCapacity.TECHNOLOGY.unique()
    #nr_of_sources = len(source_names)


    # prep inputs
    lmax_sources = np.zeros((nr_of_hours, nr_of_sources))
    util_sources = np.zeros((nr_of_hours, nr_of_sources))
    gmax_sinks = np.zeros((nr_of_hours, len(all_stream_ids)))
    cost_sinks = np.zeros((nr_of_hours, len(all_stream_ids)))
    #gmax_sources = np.array([AccumulatedNewCapacity.loc[(AccumulatedNewCapacity['YEAR'] == year) &
                    #(AccumulatedNewCapacity['TECHNOLOGY'] == x)].VALUE.to_list()[0] for x in source_names])
    #gmax_sources = np.reshape(gmax_sources, (1, nr_of_sources))
    #gmax_sources = np.repeat(gmax_sources, nr_of_hours, axis=0)
    #cost_sources = np.array(
                    #[AnnualVariableOperatingCost.loc[(AnnualVariableOperatingCost['YEAR'] == year) &
                    #(AnnualVariableOperatingCost['TECHNOLOGY'] == x)].VALUE.to_list()[0] /
                    #ProductionByTechnologyAnnual.loc[(ProductionByTechnologyAnnual['YEAR'] == year) &
                    #(ProductionByTechnologyAnnual['TECHNOLOGY'] == x)].VALUE.to_list()[0]
                    #for x in source_names])
    #cost_sources = np.reshape(cost_sources, (1, nr_of_sources))
    #cost_sources = np.repeat(cost_sources, nr_of_hours, axis=0)
    
    # # get CO2 emissions from TEO
    # AnnualTechnologyEmission = AnnualTechnologyEmission.loc[AnnualTechnologyEmission["EMISSION"] == "CO2"]
    # co2_em_sources  = np.array(
    #                             [AnnualTechnologyEmission.loc[(AnnualTechnologyEmission['YEAR'] == year) &
    #                             (AnnualTechnologyEmission['TECHNOLOGY'] == x)].VALUE.to_list()[0] /
    #                             ProductionByTechnologyAnnual.loc[(ProductionByTechnologyAnnual['YEAR'] == year) &
    #                             (ProductionByTechnologyAnnual['TECHNOLOGY'] == x)].VALUE.to_list()[0]
    #                             for x in source_names])




    ## combine in input_dict as we used to have it
    # make input dict ------------------------
    # combine source and sink inputs
    agent_ids = source_names + all_stream_ids
    gmax = np.concatenate((gmax_sources, gmax_sinks), axis=1).tolist()
    lmax = np.concatenate((lmax_sources, lmax_sinks), axis=1).tolist()
    cost = np.concatenate((cost_sources, cost_sinks), axis=1).tolist()
    util = np.concatenate((util_sources, util_sinks), axis=1).tolist()
    #co2_em = co2_em_sources.tolist() + co2_em_sinks.tolist()


    # construct input_dict
    input_dict = {
                    'md': user_input['user']['md'],
                    'horizon_basis': user_input['user']["horizon_basis"],
                    'data_profile': user_input['user']["data_profile"],
                    'recurrence': user_input['user']["recurrence"],
                    'yearly_demand_rate': user_input['user']["yearly_demand_rate"],
                    'prod_diff_option': user_input['user']["prod_diff"],
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
