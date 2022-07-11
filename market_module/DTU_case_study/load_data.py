import pandas as pd
from datetime import datetime, timedelta

def fill_missing(df,start_date,end_date):
    time_idx = pd.date_range(start=start_date,end=end_date,freq='H')
    missing_date = pd.date_range(start=start_date, end=end_date,freq='H').difference(df.index)
    df = df.reindex(time_idx)
    
    df.loc[missing_date] = df.rolling('2H').mean().loc[missing_date]
    return df

def load_data(nr_h,Year,Month,Day):
    path = "/home/linde/Documents/2019PhD/EMB3Rs/EMB3Rs-programming/market_module/DTU_case_study/Linde_data/"

    consumption_data_pre = pd.read_csv(path+'EMB3Rs_Full_Data.csv',
                                        usecols=range(1,34),
                                        parse_dates=['Hour'],
                                        index_col = ['Hour']
                                        )

    start_date = consumption_data_pre.loc[f'{Year}-{Month}-{Day}'].index[0]
    end_date = start_date + timedelta(hours=nr_h-1)                                 
    consumption_data = consumption_data_pre[start_date:end_date]
    
    df_elprice = pd.read_csv(path + 'elspot_2018-2020_DK2.csv',
                        index_col = 1, parse_dates=True)
    price_el_hourly = pd.DataFrame(df_elprice[::-1].loc[start_date:end_date,'SpotPriceDKK'],dtype=float) # time is reversed from most recent to oldest 
    price_el_hourly.index = pd.to_datetime(price_el_hourly.index)

    # check if the numbers of hours match
    if price_el_hourly.shape[0] != consumption_data.shape[0]:
        print('There is missing data')
        if price_el_hourly.shape[0] > consumption_data.shape[0]:
            consumption_data = fill_missing(consumption_data,start_date,end_date)
        else:
            print('Price_el_hourly is the problem')
            price_el_hourly = fill_missing(price_el_hourly,start_date,end_date)
 
    grid_price = pd.read_csv(path + 'df_grid_price.csv',usecols=[1])[:nr_h]
    grid_price = grid_price.set_index(consumption_data.index)
    return consumption_data, price_el_hourly, grid_price

def load_network():
    path = "/home/linde/Documents/2019PhD/EMB3Rs/EMB3Rs-programming/market_module/DTU_case_study/Linde_data/"
    nodes_name_data = pd.read_csv(path + 'Nodes_data.csv',names=['id','name'])
    buildingID = pd.read_csv(path+'BuildingID.csv',names=['Node'])   
    pipe_length = pd.read_csv(path+'pipe_data.csv')
    pipe_dir = pd.read_csv(path+'Pipe_edges.csv',names=['from','to'])
    return nodes_name_data, buildingID, pipe_length, pipe_dir