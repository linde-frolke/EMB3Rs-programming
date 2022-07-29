"""
test function for long term market, centralized offline
"""

# import own modules
from audioop import avg
from lib2to3.pytree import convert
from ...long_term.market_functions.run_longterm_market import run_longterm_market
from ...long_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs
import numpy as np
import pandas as pd
import json
from datetime import datetime
from dateutil.relativedelta import relativedelta
#from numpy import random
import xlrd #1.2.0


def test_offline():
    print('Starting offline testing...........................')
    # Open the Workbook
    workbook = xlrd.open_workbook("INPUTS.xlsx")
    # Open the worksheet
    worksheet = workbook.sheet_by_index(7)

    input_data = {}

    input_data['user'] = {
        'md': worksheet.cell_value(1,4),
        'horizon_basis': worksheet.cell_value(2,4),
        'data_profile': worksheet.cell_value(4,4),
        'recurrence': int(worksheet.cell_value(3,4)),
        'yearly_demand_rate': float(worksheet.cell_value(5,4)),
        'prod_diff': worksheet.cell_value(7,4),
        'start_datetime': worksheet.cell_value(6,4),
        'util': [worksheet.cell_value(8,4)]
    }
    #TEO data
    f = open('result.json')
    input_data['teo-module'] = json.load(f)
    f.close()

    #CF data
    f = open('cres_sinks_info.json')
    input_data['cf-module'] = json.load(f)
    f.close()

    # convert inputs to wanted scheme -------------------------------------------
    inputs_converted = convert_user_and_module_inputs(input_data=input_data)
    result_dict = run_longterm_market(input_dict=inputs_converted)

    # Serializing json
    json_object = json.dumps(result_dict, indent=4)

    # Writing to sample.json
    with open("market.json", "w") as outfile:
        outfile.write(json_object)
