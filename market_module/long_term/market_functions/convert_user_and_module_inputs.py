"""
function that converts TEO and CF inputs to the "input_dict" we were expecting
"""
import json
import numpy as np
import pandas as pd
import datetime
from datetime import datetime
from dateutil.relativedelta import relativedelta


def convert_user_and_module_inputs(input_data):
    # dummy = False 

    # user_inputs
    user_input = input_data["platform"] ### separate dictionary inside


    #Date related
    date_format = '%d-%m-%Y'
    start_date = datetime.strptime(user_input["start_datetime"], date_format)
    if user_input['horizon_basis'] == 'weeks':
        end_date = start_date + relativedelta(weeks=user_input["recurrence"])
    if user_input['horizon_basis'] == 'months':
        end_date = start_date + relativedelta(months=user_input["recurrence"])
    if user_input['horizon_basis'] == 'years':
        end_date = start_date + relativedelta(years=user_input["recurrence"])

    if user_input['data_profile'] == 'hourly':
        diff = end_date - start_date  # difference
        diff = int(diff.total_seconds()/3600) #difference in hours

    if user_input['data_profile'] == 'daily':
        diff = end_date - start_date  # difference
        diff = int(diff.total_seconds()/3600/24) #difference in days

    nr_of_hours = diff


    # # extract day month year------------------
    day, month, year = [int(x) for x in user_input["start_datetime"].split("-")]
    as_date = datetime(year=year, month=month, day=day)
    start_hourofyear = as_date.timetuple().tm_yday * 24   # start index if selecting from entire year of hourly data.
    end_hourofyear = start_hourofyear + nr_of_hours # end index if selecting from entire year of hourly data.

    # get GIS data ----------
    gis_output = input_data["gis-module"]
    nodes = [x["osmid"] for x in  gis_output["network_solution_nodes"]] 
    edges = pd.DataFrame(gis_output["network_solution_edges"])
    # gis_output["selected_agents"]  # TODO see if I can use this for something

    # get CF data
    all_sinks_info = input_data["cf-module"]["all_sinks_info"]["sinks"]

    # get TEO data
    teo_output = input_data["teo-module"]


    # convert TEO inputs ----------------------------------------------------------------------------
    AccumulatedNewCapacity = pd.json_normalize(teo_output["AccumulatedNewCapacity"])
    AccumulatedNewCapacity["YEAR"] = AccumulatedNewCapacity["YEAR"].astype(int)
    AnnualVariableOperatingCost = pd.json_normalize(teo_output["AnnualVariableOperatingCost"])
    AnnualVariableOperatingCost["YEAR"] = AnnualVariableOperatingCost["YEAR"].astype(int)
    ProductionByTechnologyAnnual = pd.json_normalize(teo_output["ProductionByTechnologyAnnual"])
    ProductionByTechnologyAnnual["YEAR"] = ProductionByTechnologyAnnual["YEAR"].astype(int)
    AnnualTechnologyEmission = pd.json_normalize(teo_output["AnnualTechnologyEmission"])
    AnnualTechnologyEmission["YEAR"] = AnnualTechnologyEmission["YEAR"].astype(int)

    # year must be one of the years that the TEO simulates for
    if not year in AccumulatedNewCapacity.YEAR.to_list():
        ValueError("User chosen 'year' must be one of the YEARs provided by TEO")
    
    # extract source names 
    source_names = AccumulatedNewCapacity.TECHNOLOGY.unique()
    nr_of_sources = len(source_names)


    # prep inputs
    lmax_sources = np.zeros((nr_of_hours, nr_of_sources))
    util_sources = np.zeros((nr_of_hours, nr_of_sources))
    gmax_sources = np.array([AccumulatedNewCapacity.loc[(AccumulatedNewCapacity['YEAR'] == year) & 
                    (AccumulatedNewCapacity['TECHNOLOGY'] == x)].VALUE.to_list()[0] for x in source_names])
    gmax_sources = np.reshape(gmax_sources, (1, nr_of_sources))
    gmax_sources = np.repeat(gmax_sources, nr_of_hours, axis=0)
    cost_sources = np.array(
                    [AnnualVariableOperatingCost.loc[(AnnualVariableOperatingCost['YEAR'] == year) & 
                    (AnnualVariableOperatingCost['TECHNOLOGY'] == x)].VALUE.to_list()[0] /
                    ProductionByTechnologyAnnual.loc[(ProductionByTechnologyAnnual['YEAR'] == year) & 
                    (ProductionByTechnologyAnnual['TECHNOLOGY'] == x)].VALUE.to_list()[0]                 
                    for x in source_names])
    cost_sources = np.reshape(cost_sources, (1, nr_of_sources))
    cost_sources = np.repeat(cost_sources, nr_of_hours, axis=0)
    
    # get CO2 emissions from TEO 
    AnnualTechnologyEmission = AnnualTechnologyEmission.loc[AnnualTechnologyEmission["EMISSION"] == "CO2"]
    co2_em_sources  = np.array(
                                [AnnualTechnologyEmission.loc[(AnnualTechnologyEmission['YEAR'] == year) & 
                                (AnnualTechnologyEmission['TECHNOLOGY'] == x)].VALUE.to_list()[0] /
                                ProductionByTechnologyAnnual.loc[(ProductionByTechnologyAnnual['YEAR'] == year) & 
                                (ProductionByTechnologyAnnual['TECHNOLOGY'] == x)].VALUE.to_list()[0]                 
                                for x in source_names])


    # convert CF inputs -----------------------------------------------------------------------------
    # get some sink info 
    nr_of_sinks = len(all_sinks_info)
    sink_ids = []
    #sink_locs = []
    map_sink_streams = []

    for i in range(nr_of_sinks):
        sink_ids += [all_sinks_info[i]["sink_id"]]
        #sink_locs += [all_sinks_info[i]["location"]]
        map_sink_streams += [[streams["stream_id"] for streams in all_sinks_info[i]["streams"]]]

    # get some stream info 
    all_stream_ids = [y for x in map_sink_streams for y in x]
    nr_of_streams = len(all_stream_ids)
    timesteps = len(all_sinks_info[0]["streams"][0]["hourly_stream_capacity"])

    # prep inputs
    lmax_sinks = np.zeros((nr_of_hours, nr_of_streams))
    counter = 0 
    for j in range(nr_of_sinks):
        for i in range(len(map_sink_streams[j])):
            # if dummy:
            #     all_sinks_info[j]["streams"][i]["hourly_stream_capacity"] = (
            #         all_sinks_info[j]["streams"][i]["hourly_stream_capacity"] * 400*20)[1:8784]
            # 

            lmax_sinks[:, counter] = all_sinks_info[j]["streams"][i]["hourly_stream_capacity"][start_hourofyear:end_hourofyear]
            counter += 1

    gmax_sinks = np.zeros(np.shape(lmax_sinks))
    cost_sinks = np.zeros(np.shape(lmax_sinks))
    util_sinks = np.array([user_input["util"][x] for x in all_stream_ids]).transpose()
    co2_em_sinks = np.zeros(nr_of_sinks)

    ## combine in input_dict as we used to have it
    # make input dict ------------------------
    # combine source and sink inputs
    agent_ids = list(source_names) + all_stream_ids
    gmax = np.concatenate((gmax_sources, gmax_sinks), axis=1).tolist()
    lmax = np.concatenate((lmax_sources, lmax_sinks), axis=1).tolist()
    cost = np.concatenate((cost_sources, cost_sinks), axis=1).tolist()
    util = np.concatenate((util_sources, util_sinks), axis=1).tolist()
    co2_em = co2_em_sources.tolist() + co2_em_sinks.tolist()

    # construct input_dict
    input_dict = {
                    'md': user_input['md'],  # other options are  'p2p' or 'community'
                    'horizon_basis': user_input["horizon_basis"],
                    'data_profile': user_input["data_profile"],
                    'recurrence': user_input["recurrence"],
                    'yearly_demand_rate': user_input["yearly_demand_rate"],
                    'prod_diff_option': user_input["prod_diff"],
                    'agent_ids': agent_ids,
                    'gmax': gmax,
                    'lmax': lmax,
                    'cost': cost,
                    'util': util,
                    'start_datetime': "31-01-2002",
                    'co2_emissions': co2_em,  # allowed values are 'none' or array of size (nr_of_agents)
                    'gis_data': gis_output["res_sources_sinks"],
                    'nodes': nodes,
                    'edges': edges
                    }

    return input_dict
