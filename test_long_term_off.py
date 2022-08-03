### MODULE-CODE [BEGIN]
from cProfile import label
from unittest import result
import pandas as pd
import matplotlib.pyplot as plt, mpld3
from market_module.long_term.market_functions.run_longterm_market import run_longterm_market
from market_module.long_term.market_functions.convert_user_and_module_inputs import convert_user_and_module_inputs
from market_module.long_term.plotting_processing_functions.outputs_to_html import output_to_html, output_to_html_no_index, output_to_html_no_index_transpose, output_to_html_list
import warnings
from datetime import datetime
import json
warnings.filterwarnings("ignore")




input_data = {}

input_data['user'] = {'md': 'centralized', 'horizon_basis': 'years', 'recurrence': 1,
                      'data_profile': 'hourly', 'yearly_demand_rate': 0.05,
                      'start_datetime': datetime(2018, 1, 1, 0, 0),
                      'prod_diff_option': 'noPref',
                      'util': [40.5, 500, 600, 1500, 3000, 50, 4000, 60, 700, 7000, 35, 66, 44, 888]}

# TEO data
f = open('teo_results.json')
input_data['teo-module'] = json.load(f)
f.close()

# CF data
f = open('convert_sinks_results.json')
input_data['cf-module'] = json.load(f)
f.close()

# convert inputs to wanted scheme -------------------------------------------
inputs_converted = convert_user_and_module_inputs(input_data=input_data)
result_dict = run_longterm_market(input_dict=inputs_converted)

# # Serializing json
# json_object = json.dumps(result_dict, indent=4)
#
# # Writing to sample.json
# with open("market_off.json", "w") as outfile:
#     outfile.write(json_object)

# #%-------------------------
# # CF data
# from market_module.long_term.plotting_processing_functions.outputs_to_html import output_to_html, output_to_html_no_index, output_to_html_no_index_transpose, output_to_html_list
# import warnings
# import json
# f = open('market_off.json')
# result_dict = json.load(f)
# f.close()

## convert outputs to html to put in report
df_Gn, df_Ln, df_Pn, df_set, df_ag_op_cost = [output_to_html(result_dict[x], filter="sum") for x in
                                        ["Gn", "Ln", "Pn", "settlement", "agent_operational_cost"]]

#Results with different format
df_spm = output_to_html_no_index(result_dict["SPM"])
df_adg = output_to_html_no_index(result_dict["ADG"])
df_qoe = output_to_html_no_index_transpose(result_dict["QoE"])
df_social_w = output_to_html_list(result_dict['social_welfare_h'], filter='mean')
df_shadow_price = output_to_html_list(result_dict['shadow_price'], filter='mean')


### MODULE-CODE [END]


### REPORT_RENDERING_CODE [BEGIN]
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader('asset'),
    autoescape=False
)

template = env.get_template('index.longtermtemplate.html')
template_content = template.render(df_Gn=df_Gn, df_Ln=df_Ln, df_Pn=df_Pn, df_set=df_set, df_ag_op_cost=df_ag_op_cost,
                                   df_spm=df_spm, df_adg=df_adg, df_qoe=df_qoe, df_social_w=df_social_w,
                                   df_shadow_price=df_shadow_price)


f = open("test_long_term_off.html", "w")
f.write(template_content)
f.close()
