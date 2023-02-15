import pandas as pd
import os
import numpy as np
from functools import reduce
from sys import platform 
import matplotlib.pyplot as plt
prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

def sbrl_preprocess():
    phenotypic_data = pd.read_excel(prefix+'/Shared drives/SBRL/PhenotypicDataConsolidated.xlsx')
    phenotypic_data_no_null = phenotypic_data[~phenotypic_data.Confirmed_Final_Identification.isna()]
    phenotypic_data_no_null = phenotypic_data_no_null.assign(lower_Confirmed_Final_Identification = phenotypic_data_no_null['Confirmed_Final_Identification'].str.lower())
    # remove rows if the IDs contain the following
    phenotypic_data_confirmed = phenotypic_data_no_null[~phenotypic_data_no_null['lower_Confirmed_Final_Identification'].str.contains('tentative|unidentified|contaminated')]
    # remove ID == 'non-viable'
    phenotypic_data_confirmed = phenotypic_data_confirmed[phenotypic_data_confirmed['lower_Confirmed_Final_Identification'] != '(non-viable)']
    # remove "transferred" in the ID
    phenotypic_data_confirmed = phenotypic_data_confirmed.assign(final_id_clean =
                                                                 phenotypic_data_confirmed['lower_Confirmed_Final_Identification'].str.replace('\(transferred\)', '').str.strip())
    # remove "non-viable" in the ID
    phenotypic_data_confirmed['final_id_clean'] = phenotypic_data_confirmed['final_id_clean'].str.replace('\(non-viable\)', '').str.strip()
    # copy the genus to final_id_clean if it's empty
    phenotypic_data_confirmed.loc[phenotypic_data_confirmed.final_id_clean == '', 'final_id_clean'] = phenotypic_data_confirmed.loc[phenotypic_data_confirmed.final_id_clean == '', 'Genus'].str.lower()
    return phenotypic_data_confirmed.reset_index(drop = True)

def get_available_data(cdc_data, cols_to_use, value_options, plot = True):
    '''Convert unexpected values in the data to NA'''
    valid_cnt = []
    valid_df = []
    for col in cols_to_use:
        valid_temp = cdc_data[cdc_data[col].isin(value_options)][col].reset_index()
        valid_df.append(valid_temp)
        valid_cnt.append(len(valid_temp)/len(cdc_data))
    joined_valid_df = reduce(lambda left, right: pd.merge(left, right, on = 'index', how = 'outer'), valid_df)
    valid_cnt_df = pd.DataFrame(valid_cnt, index = cols_to_use)
    if plot == True:
        plot_available_data(valid_cnt_df)
    return joined_valid_df

def plot_available_data(valid_cnt_df):
    '''plot % of data with expected values in each test'''
    plt.figure(figsize = (1, len(valid_cnt_df)/1.5))
    sns.heatmap(valid_cnt_df, cmap = 'Reds', center = 0.5, vmin = 0, vmax = 1, annot = True)
    plt.yticks(rotation = 0)
    plt.xticks([])
    plt.title('Percentage of available data')


def join_valid_test_df(df_list, original_data):
    '''After getting data with valid entries from get_available_data(), join the dfs together with this function'''
    original_data['index'] = original_data.index
    joined_valid = reduce(lambda left, right: pd.merge(left, right, on = 'index', how = 'outer'), df_list)
    joined_valid_w_id = pd.merge(joined_valid, original_data[['index', 'final_id_clean']], left_on = 'index', right_on = 'index')
    joined_valid_w_id.drop(columns = 'index', inplace = True)
    return joined_valid_w_id