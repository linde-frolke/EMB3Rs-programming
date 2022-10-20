## make storage work

import numpy as np
import pandas as pd
# import xlrd
from dateutil.relativedelta import relativedelta
from dateutil.parser import parse
import datetime
import json 
from math import ceil

f1 = open("/home/linde/Downloads/market-module-long-term-input.json")
input_data = json.load(f1)

# user_inputs
user_input = input_data ### separate dictionary inside

# Date related
datetime_date = user_input['platform']['start_datetime']
start_date = parse(datetime_date)
start_date_str = start_date.strftime('%d-%m-%Y')

if user_input['platform']['horizon_basis'] == 'weeks':
    end_date = start_date + relativedelta(weeks=user_input['platform']["recurrence"])
if user_input['platform']['horizon_basis'] == 'months':
    end_date = start_date + relativedelta(months=user_input['platform']["recurrence"])
if user_input['platform']['horizon_basis'] == 'years':
    #always sending just one year of hourly data
    end_date = start_date + relativedelta(years=1)

if user_input['platform']['data_profile'] == 'hourly':
    diff = end_date - start_date  # difference
    diff = int(diff.total_seconds()/3600) #difference in hours

if user_input['platform']['data_profile'] == 'daily':
    diff = end_date - start_date  # difference
    # always sending just one year of hourly data
    diff = int(diff.total_seconds()/3600) #difference in hours

nr_of_hours = diff

def date_range(start, end):
    r = (end+datetime.timedelta(days=1)-start).days
    return [start+datetime.timedelta(days=i) for i in range(r)]


# create a list stating for each timestep of the period what year it is.
dates = []
d = start_date
while d < end_date:
    dates.append(d)
    d += datetime.timedelta(hours=1)
year_ = [x.year for x in dates]
set(year_)

# Opening JSON file TEO WITH STORAGE
f = open("/home/linde/Documents/2019PhD/EMB3Rs/module_integration/TEOoutputs_with_storage.json")
teo_output = json.load(f)
teo_output.keys()


storage_data_TEO = teo_output["AccumulatedNewStorageCapacity"]

# extract the needed storage data
storage_df = pd.DataFrame(storage_data_TEO)
storage_df.YEAR = storage_df["YEAR"].astype(int)
for year in set(year_):
    if not year in set(storage_df.YEAR):
        raise RuntimeError("The TEO data for storage capacity does not cover the selected simulation time. ")
# nr and names
nr_of_storage = len(storage_data_TEO)
storage_names = list(set(storage_df.STORAGE))

storage_name = "tankstorage"
(storage_df.STORAGE == storage_name) & (storage_df.YEAR == 2023)

# capacity per year 
storage_capacity_per_timestep = {}
if nr_of_storage > 0:
    for storage_name in storage_names:
        capacity_per_time = [storage_df.VALUE[(storage_df.STORAGE == storage_name) & (storage_df.YEAR == year_nr)].to_numpy().item() for year_nr in year_]
        storage_capacity_per_timestep[storage_name] = capacity_per_time

set(storage_capacity_per_timestep.keys()) == set(storage_names)

len(storage_capacity_per_timestep["tankstorage"])

stor_capacity_array = np.array([storage_capacity_per_timestep[stor] for stor in storage_names]).T.tolist()


# timesteps 
t = 8700
h_per_iter = 24 
nr_of_iter = ceil(t / h_per_iter)
iter_days = range(nr_of_iter)
h_on_last = t - (nr_of_iter - 1)*h_per_iter 

for iter in iter_days:
    # set the number of timesteps in this iteration
    if iter == (nr_of_iter - 1):
        nr_of_timesteps = h_on_last
    else:
        nr_of_timesteps = h_per_iter
        
iter = 0
selected_timesteps = range(iter*h_per_iter, (iter +1)*h_per_iter)
[i for i in selected_timesteps]