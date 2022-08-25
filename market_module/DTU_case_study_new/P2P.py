from market_module.short_term.market_functions.run_shortterm_market import run_shortterm_market
import numpy as np
import pandas as pd
from datetime import timedelta
from itertools import chain

from market_module.DTU_case_study_new.save_tocsv import save_topickle, CaseStudyData
from market_module.DTU_case_study_new.prep_inputs import prep_inputs, make_gis_data

# p2p Base
##---------------------------
one_year_idx, nr_h, agent_ids, g_max, l_max, cost, utility, time_range = prep_inputs()

# monthly average
shadow_price_year = []
Pn_year = []
Gn_year = []
Ln_year = []

sw_year = []
settlement_year = []
Gn_revenue_year = []
Ln_revenue_year = []
Tnm_year = []

for i,date in enumerate(one_year_idx,1):
	print('-'*80)
	print(date)
	print('-'*80)
	
	g_max_list = g_max[24*(i-1):24*i,:].tolist()
	l_max_list = l_max[24*(i-1):24*i,:].tolist()

	cost_list = cost[24*(i-1):24*i,:].tolist()
	utility_list = utility[24*(i-1):24*i,:].tolist()

	input_dict = {
		'md': 'p2p',  # other options are  'p2p' or 'community'
		'nr_of_hours': nr_h,
		'offer_type': 'simple',
		'prod_diff': 'noPref',
		'network': 'none',
		'el_dependent': 'false',
		'el_price': 'none', # not list but array 
		'agent_ids': agent_ids,
		'objective': 'none', 
		'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
		'gmax': g_max_list, 
		'lmax': l_max_list,
		'cost': cost_list, 
		'util': utility_list, 
		'co2_emissions': 'none',
		'is_in_community': 'none', 
		'block_offer': 'none', 
		'is_chp': 'none', 
		'chp_pars': 'none',  
		'gis_data': 'none',  
		'nodes': 'none', 
		'edges': 'none',
		}
	# 
	results = run_shortterm_market(input_dict=input_dict)
	# 
	shadow_price = [pd.DataFrame(results['shadow_price'][i]) for i in range(len(results['shadow_price']))]

	Pn = pd.DataFrame.from_dict(results['Pn'])
	Gn = pd.DataFrame.from_dict(results['Gn'])
	Ln = pd.DataFrame.from_dict(results['Ln'])
	Tnm = [pd.DataFrame.from_dict(results['Tnm'][i]) for i in range(len(results["Tnm"]))]
	sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
	settlement = pd.DataFrame.from_dict(results['settlement'])

	Gn_revenue = Pn * 0 # this is not working now
	Ln_revenue = Ln * 0
	# for t in range(24):
	# 	for agent in Pn.columns:
	# 		Gn_revenue[agent][t] = sum([shadow_price[t].loc[agent, agent2] * Snm[t].loc[agent, agent2] 
	# 										for agent2 in Pn.columns])
	# Ln_revenue = pd.DataFrame(shadow_price.values*Ln.values, columns=Ln.columns, index=Ln.index)

	shadow_price_year += shadow_price
	Pn_year.append(Pn)
	Gn_year.append(Gn)
	Ln_year.append(Ln)
	sw_year.append(sw)
	settlement_year.append(settlement)
	Gn_revenue_year.append(Gn_revenue)
	Ln_revenue_year.append(Ln_revenue)
	Tnm_year += Tnm


newest = list(chain.from_iterable(shadow_price_year))
Tnm_new = list(chain.from_iterable(Tnm))
len(newest)

# Save and export dataframes to CSV
model_name = 'P2P_base'
p2p_data = CaseStudyData(model_name,time_range,newest,Pn_year, 
                            Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year, p2p=True, 
							Tnm=Tnm_new)
save_topickle(model_name=model_name, casedata=p2p_data)


## p2p WITH preferences
print("running p2p with distance preference ")
##---------------------------
one_year_idx, nr_h, agent_ids, g_max, l_max, cost, utility, time_range = prep_inputs()
gis_data = make_gis_data(agent_ids=agent_ids)
test = pd.DataFrame(gis_data)

# monthly average
shadow_price_year = []
Pn_year = []
Gn_year = []
Ln_year = []

sw_year = []
settlement_year = []
Gn_revenue_year = []
Ln_revenue_year = []
Tnm_year = []

for i,date in enumerate(one_year_idx,1):
	print('-'*80)
	print(date)
	print('-'*80)
	
	g_max_list = g_max[24*(i-1):24*i,:].tolist()
	l_max_list = l_max[24*(i-1):24*i,:].tolist()

	cost_list = cost[24*(i-1):24*i,:].tolist()
	utility_list = utility[24*(i-1):24*i,:].tolist()

	input_dict = {
		'md': 'p2p',  # other options are  'p2p' or 'community'
		'nr_of_hours': nr_h,
		'offer_type': 'simple',
		'prod_diff': 'networkDistance',
		'network': 'none',
		'el_dependent': 'false',
		'el_price': 'none', # not list but array 
		'agent_ids': agent_ids,
		'objective': 'none', 
		'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
		'gmax': g_max_list, 
		'lmax': l_max_list,
		'cost': cost_list, 
		'util': utility_list, 
		'co2_emissions': 'none',
		'is_in_community': 'none', 
		'block_offer': 'none', 
		'is_chp': 'none', 
		'chp_pars': 'none',  
		'gis_data': gis_data,  
		'nodes': 'none', 
		'edges': 'none',
		}
	# 
	print("running market")
	results = run_shortterm_market(input_dict=input_dict)
	# 
	print("error is after this")
	shadow_price = [pd.DataFrame(results['shadow_price'][i]) for i in range(len(results['shadow_price']))]

	Pn = pd.DataFrame.from_dict(results['Pn'])
	Gn = pd.DataFrame.from_dict(results['Gn'])
	Ln = pd.DataFrame.from_dict(results['Ln'])
	Tnm = [pd.DataFrame.from_dict(results['Tnm'][i]) for i in range(len(results["Tnm"]))]
	print("error is after this 2")
	sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
	settlement = pd.DataFrame.from_dict(results['settlement'])

	Gn_revenue = Pn * 0
	Ln_revenue = Ln * 0
	# for t in range(24):
	# 	for agent in Pn.columns:
	# 		Gn_revenue[agent][t] = sum([shadow_price[t].loc[agent, agent2] * Snm[t].loc[agent, agent2] 
	# 										for agent2 in Pn.columns])
	# Ln_revenue = pd.DataFrame(shadow_price.values*Ln.values, columns=Ln.columns, index=Ln.index)

	shadow_price_year += shadow_price
	Pn_year.append(Pn)
	Gn_year.append(Gn)
	Ln_year.append(Ln)
	sw_year.append(sw)
	settlement_year.append(settlement)
	Gn_revenue_year.append(Gn_revenue)
	Ln_revenue_year.append(Ln_revenue)
	Tnm_year += Tnm


from itertools import chain

newest = list(chain.from_iterable(shadow_price_year))
Tnm_new = list(chain.from_iterable(Tnm))
len(newest)

# Save and export dataframes to CSV
model_name = 'P2P_base'
p2p_data = CaseStudyData(model_name,time_range,newest,Pn_year, 
                            Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year, p2p=True, 
							Tnm=Tnm_new)
save_topickle(model_name=model_name, casedata=p2p_data)

# ## p2p WITH Energy budget
# print("running p2p with distance preference and budget balance")
# # adapt l_max
# l_max_EB = l_max * 2

# # to store monthly average
# shadow_price_year = []
# Pn_year = []
# Gn_year = []
# Ln_year = []

# sw_year = []
# settlement_year = []
# Gn_revenue_year = []
# Ln_revenue_year = []
# for i,date in enumerate(one_year_idx,1):
# 	print('-'*80)
# 	print(date)
# 	print('-'*80)
	
# 	g_max_list = g_max[24*(i-1):24*i,:].tolist()
# 	l_max_list = l_max_EB[24*(i-1):24*i,:].tolist()

# 	cost_list = cost[24*(i-1):24*i,:].tolist()
# 	utility_list = utility[24*(i-1):24*i,:].tolist()

# 	input_dict = {
# 		'md': 'p2p',  # other options are  'p2p' or 'community'
# 		'nr_of_hours': nr_h,
# 		'offer_type': 'energyBudget',
# 		'prod_diff': 'noPref',
# 		'network': 'none',
# 		'el_dependent': 'false',
# 		'el_price': 'none', # not list but array 
# 		'agent_ids': agent_ids,
# 		'objective': 'none', 
# 		'community_settings': {'g_peak': 'none', 'g_exp': 'none', 'g_imp': 'none'}, 
# 		'gmax': g_max_list, 
# 		'lmax': l_max_list,
# 		'cost': cost_list, 
# 		'util': utility_list, 
# 		'co2_emissions': 'none',
# 		'is_in_community': 'none', 
# 		'block_offer': 'none', 
# 		'is_chp': 'none', 
# 		'chp_pars': 'none',  
# 		'gis_data': 'none',  
# 		'nodes': 'none', 
# 		'edges': 'none',
# 		}

# 	results = run_shortterm_market(input_dict=input_dict)

# 	shadow_price = pd.DataFrame(results['shadow_price']['uniform price'])

# 	Pn = pd.DataFrame.from_dict(results['Pn'])
# 	Gn = pd.DataFrame.from_dict(results['Gn'])
# 	Ln = pd.DataFrame.from_dict(results['Ln'])

# 	sw = pd.DataFrame.from_dict((results['social_welfare_h']['Social Welfare']))
# 	settlement = pd.DataFrame.from_dict(results['settlement'])

# 	Gn_revenue = pd.DataFrame(shadow_price.values*Gn.values, columns=Gn.columns, index=Gn.index)
# 	Ln_revenue = pd.DataFrame(shadow_price.values*Ln.values, columns=Ln.columns, index=Ln.index)

# 	shadow_price_year.append(shadow_price)
# 	Pn_year.append(Pn)
# 	Gn_year.append(Gn)
# 	Ln_year.append(Ln)
# 	sw_year.append(sw)
# 	settlement_year.append(settlement)
# 	Gn_revenue_year.append(Gn_revenue)
# 	Ln_revenue_year.append(Ln_revenue)

# # Save and export dataframes to CSV
# model_name = 'p2p_EB'
# p2p_data_EB = CaseStudyData(model_name,time_range,shadow_price_year,Pn_year, 
#                             Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year)

# save_topickle(model_name=model_name, casedata=p2p_data_EB)
# """
