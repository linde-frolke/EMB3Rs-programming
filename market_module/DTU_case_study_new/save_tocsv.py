import pandas as pd
import pickle
def save_tocsv(model_name,time_range,uniform_price_year,Pn_year,Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year):
    
    # Path to save the CSV files
    path = "market_module/DTU_case_study/Data_new/"
    
    df_uniform_price_year = pd.concat(uniform_price_year).head(len(time_range)).set_index(time_range)
    df_Pn_year = pd.concat(Pn_year).head(len(time_range)).set_index(time_range)
    df_Gn_year = pd.concat(Gn_year).head(len(time_range)).set_index(time_range)
    df_Ln_year = pd.concat(Ln_year).head(len(time_range)).set_index(time_range)
    df_sw_year = pd.concat(sw_year).head(len(time_range)).set_index(time_range)
    df_settlement_year = pd.concat(settlement_year).head(len(time_range)).set_index(time_range)
    df_Gn_rev_year = pd.concat(Gn_revenue_year).head(len(time_range)).set_index(time_range)
    df_Ln_rev_year = pd.concat(Ln_revenue_year).head(len(time_range)).set_index(time_range)

    # Export to CSV
    df_uniform_price_year.to_csv(path+f'df_uniform_price_year{model_name}.csv')
    df_Pn_year.to_csv(path+f'df_Pn_year{model_name}.csv')
    df_Gn_year.to_csv(path+f'df_Gn_year{model_name}.csv')
    df_Ln_year.to_csv(path+f'df_Ln_year{model_name}.csv')
    df_sw_year.to_csv(path+f'df_sw_year{model_name}.csv')
    df_settlement_year.to_csv(path+f'df_settlement_year{model_name}.csv')
    df_Gn_rev_year.to_csv(path+f'df_Gn_revenue_year{model_name}.csv')
    df_Ln_rev_year.to_csv(path+f'df_Ln_revenue_year{model_name}.csv')


class CaseStudyData:
    def __init__(self, model_name,time_range, uniform_price_year,
                Pn_year,Gn_year,Ln_year,sw_year,settlement_year,Gn_revenue_year,Ln_revenue_year, p2p=False,
                Tnm=None):
        self.name = model_name
        self.timerange = time_range
        if p2p:
            self.price = uniform_price_year
            self.Tnm = Tnm
        else:
            self.price = pd.concat(uniform_price_year).head(len(time_range)).set_index(time_range)
        self.Pn = pd.concat(Pn_year).head(len(time_range)).set_index(time_range)
        self.Gn = pd.concat(Gn_year).head(len(time_range)).set_index(time_range)
        self.Ln = pd.concat(Ln_year).head(len(time_range)).set_index(time_range)
        self.SW = pd.concat(sw_year).head(len(time_range)).set_index(time_range)
        self.settlement = pd.concat(settlement_year).head(len(time_range)).set_index(time_range)
        self.Gn_rev = pd.concat(Gn_revenue_year).head(len(time_range)).set_index(time_range)
        self.Ln_rev = pd.concat(Ln_revenue_year).head(len(time_range)).set_index(time_range)


def save_topickle(model_name, casedata):
    path = "market_module/DTU_case_study_new/output_data/"
    
    with open(path + model_name + '.pickle', 'wb') as handle:
        pickle.dump(casedata, handle)

def load_frompickle(model_name):
    path = "market_module/DTU_case_study_new/output_data/"
    filehandler = open(path + model_name + '.pickle', 'rb') 
    casedata = pickle.load(filehandler)
    return casedata


