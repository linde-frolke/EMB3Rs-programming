# Case study -- pool 
from market_module.DTU_case_study.load_case_study_inputs import load_data_from_files
from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market
import numpy as np

# settings
start_day = 40
nr_h = 48
grid_max_import_kWh_h = 10**5

# compute start hour of year
start_h = start_day*24
end_h = start_h + nr_h

# load data from csv files
df_grid_price, df_el_price, df_cons_profiles, df_sm_computations = load_data_from_files()

# create inputs in the right format
agent_ids = ["supermarket", "grid"] + list(df_cons_profiles.columns)[2:(len(df_cons_profiles.columns) - 2)] 
nr_agents = len(agent_ids)

# generate gmax and lmax
gmax = np.zeros((nr_h, nr_agents))
gmax[:, agent_ids.index("supermarket")] = df_sm_computations.Qdot_DH[start_h:end_h]
gmax[:, agent_ids.index("grid")] = [grid_max_import_kWh_h] * nr_h
gmax = gmax.tolist()

lmax = np.zeros((nr_h, nr_agents))
lmax[:, 2:] = df_cons_profiles.iloc[start_h:end_h, 2:32]
lmax = lmax.tolist()

# generate cost and utility
cost = np.zeros((nr_h, nr_agents))
cost[:, agent_ids.index("supermarket")] = df_el_price.SpotPriceEUR.to_numpy()[start_h:end_h]
cost[:, agent_ids.index("grid")] = df_grid_price.grid_price.to_numpy()[start_h:end_h]
max_cost = cost.max()
cost = cost.tolist()

util = np.zeros((nr_h, nr_agents))
util[:, 2:] = np.ones(util[:, 2:].shape) * (max_cost + 1) # set it to be higher than the max cost
util = util.tolist()

# put it together
input_dict = {'md': 'pool',  # other options are  'p2p' or 'community'
              'nr_of_hours': nr_h,
                'offer_type': 'simple',
                'prod_diff': 'noPref',
                'network': 'none',
                'el_dependent': 'false',
                'el_price': 'none',
                'agent_ids': agent_ids,
                'objective': 'none', 
                'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
                'gmax': gmax, 
                'lmax': lmax, 
                'cost': cost, 
                'util': util, 
                'co2_emissions': 'none',  
                'is_in_community': 'none', 
                'block_offer': 'none', 
                'is_chp': 'none', 
                'chp_pars': 'none',  
                'gis_data': 'none',  
                'nodes' : "none", 
                "edges" : "none"
                  }

result_dict = run_shortterm_market(input_dict=input_dict) 

result_dict