### MODULE-CODE [BEGIN]
from cProfile import label
from unittest import result
import pandas as pd
import matplotlib.pyplot as plt, mpld3
from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market
from market_module.short_term.plotting_processing_functions.outputs_to_html import output_to_html, output_to_plot
import warnings
warnings.filterwarnings("ignore")

# run simple pool market 
input_dict = {#'sim_name': 'test_pool',
                'md': 'pool',  # other options are  'p2p' or 'community'
                'nr_of_hours': 12,
                'offer_type': 'simple',
                'prod_diff': 'noPref',
                'network': 'none',
                'el_dependent': 'false',  # can be false or true
                'el_price': 'none',
                'agent_ids': ["prosumer_1",
                            "prosumer_2", "consumer_1", "producer_1"],
                'agent_types': ["prosumer", "prosumer", "consumer", "producer"],
                'objective': 'none',  # objective for community
                'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'},
                'gmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                'gmax': [[1, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3], [0, 0, 0, 0], [1, 1, 0, 1],
                        [2, 3, 0, 1], [4, 2, 0, 5], [3, 4, 0, 4], [1, 5, 0, 3],
                        [0, 0, 0, 0], [1, 1, 0, 1], [2, 3, 0, 1]],
                'lmin': [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0],
                        [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
                'lmax': [[2, 2, 1, 0], [2, 1, 0, 0], [1, 2, 1, 0], [3, 0, 2, 0], [1, 1, 4, 0],
                        [2, 3, 3, 0], [4, 2, 1, 0], [3, 4, 2, 0], [1, 5, 3, 0], [0, 0, 5, 0],
                        [1, 1, 3, 0], [2, 3, 1, 0]],
                'cost': [[24, 25, 45, 30], [31, 24, 0, 24], [18, 19, 0, 32], [0, 0, 0, 0],
                        [20, 25, 0, 18], [25, 31, 0, 19], [24, 27, 0, 22], [32, 31, 0, 19],
                        [15, 25, 0, 31], [0, 0, 0, 0], [19, 20, 0, 21], [22, 33, 0, 17]],
                'util': [[40, 42, 35, 0], [45, 50, 40, 0], [55, 36, 45, 0], [44, 34, 43, 0],
                        [34, 44, 55, 0], [29, 33, 45, 0], [40, 55, 33, 0],
                        [33, 42, 38, 0], [24, 55, 35, 0], [25, 35, 51, 0], [19, 43, 45, 0], [34, 55, 19, 0]],
                'co2_emissions': 'none',  # allowed values are 'none' or array of size (nr_of_agents)
                'is_in_community': 'none',  # allowed values are 'none' or boolean array of size (nr_of_agents)
                'block_offer': 'none',
                'is_chp': 'none',  # allowed values are 'none' or a list with ids of agents that are CHPs
                'chp_pars': 'none',
                'gis_data': 'none',
                'nodes' : ["prosumer_1", "prosumer_2", "consumer_1", "producer_1"],
                'edges' : [("producer_1","consumer_1"), ("producer_1","prosumer_1"),
                            ("prosumer_1","prosumer_2"), ]
                }

result_dict = run_shortterm_market(input_dict=input_dict)


## convert outputs to html to put in report

df_Gn, df_Ln, df_Pn, df_set = [output_to_html(result_dict[x]) for x in 
                                    ["Gn", "Ln", "Pn", "settlement"]]

# make plots
ylab = {"Gn" : "generation", "Ln" : "load", "Pn" : "net power injection", 
        "settlement" : "settlement"}
plt_Gn, plt_Ln, plt_Pn, plt_set = [output_to_plot(result_dict[x], ylab[x]) for x in 
                                    ["Gn", "Ln", "Pn", "settlement"]]

### MODULE-CODE [END]


### REPORT_RENDERING_CODE [BEGIN]
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape

env = Environment(
    loader=FileSystemLoader('asset'),
    autoescape=False
)

template = env.get_template('index.shorttermtemplate.html')
template_content = template.render(df_Gn=df_Gn, df_Ln=df_Ln, df_Pn=df_Pn, df_set=df_set, 
                                    plt_Gn=plt_Gn, plt_Ln=plt_Ln, plt_Pn=plt_Pn, plt_set=plt_set)


f = open("index.html", "w")
f.write(template_content)
f.close()

output = {
    "every" : "thing",
    "else" : "yes",
    "report" : template_content
}