# case study inputs for DTU -- Nordhavn ELN 

import numpy as np 
import pandas as pd


def load_data_from_files():
    data_folder = "C:/Users/hyung/Documents/Desktop/Student Job/Data/"

    df_grid_price = pd.read_csv(data_folder + 'copenhagen_2019_DH_grid_price.csv')

    df_el_price = pd.read_csv(data_folder + 'elspot_2018-2020_DK2.csv')[["HourUTC", "HourDK", "SpotPriceEUR"]]

    df_cons_profiles = pd.read_csv(data_folder + 'EMB3Rs_Full_Data.csv')

    df_sm_computations = pd.read_csv(data_folder + "Wiebke_calculations_supermarket_directprofile.csv")[["T_amb", "Qdot_DH"]] ## Qdot in kW

    ## create supermarket profile given ambient temperature
    return df_grid_price, df_el_price, df_cons_profiles, df_sm_computations

