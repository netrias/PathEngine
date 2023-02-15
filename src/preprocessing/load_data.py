''''
This file will be used to help us load all of our data from the different assays
'''
import pandas as pd
import os
from functools import reduce
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None
from sys import platform 
prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'
import FlowCal

def strain_mapping(file):
    strain_file = pd.read_excel(file).fillna(0)
    strain_file['Name'] = strain_file['Name'].apply(str).apply(str.lower)
    return strain_file
def get_mapping_df():
    mapping_df = pd.read_excel(prefix+'/Shared drives/STTR_Netrias/References/strain_name_mapping.xlsx')
    mapping_df['Name'] = mapping_df['Name'].apply(str).apply(str.lower)
    return mapping_df


def adherence_data(dir):
    '''
    Load adherence data from a directory and transform it into the dataframe needed for the model
    dir: A directory where adherence data are
    strain_mapping_data: Excel file for mapping strain names
    return: Dataframe ready for analysis for the adherence model
    '''
    strain_mapping_data = get_mapping_df()
    file_paths = [dir + '/' + file for file in os.listdir(dir)]
    dfs = []
    for file in file_paths:
        adh_df = pd.read_csv(file)
        dfs.append(adh_df)
    # reduce function to merge all the dfs
    merged_adh_df = reduce(lambda left, right: pd.merge(left, right, how = 'outer'), dfs)
    merged_adh_df['Organism'] = merged_adh_df['Organism'].str.lower()
    # merge adherence data with strain mapping file
    adh_labeled = pd.merge(merged_adh_df, strain_mapping_data, how = 'inner',
                           left_on = 'Organism', right_on = 'Name')
    adh_labeled.drop(
        ['#', 'Size', 'Area', 'Mean[Deconvolved[Tsf[DAPI 377,447]]]', 'Area[Deconvolved[Tsf[GFP 469,525]]]',
         'Mean[Deconvolved[Tsf[GFP 469,525]]]', 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]',
         'Area_2[Deconvolved[Tsf[GFP 469,525]]]', 'Mean_2[Deconvolved[Tsf[GFP 469,525]]]'], axis=1, inplace=True)

    adh_labeled['cnt'] = 1


    # selecting columns from adh_labeled to groupby for counting
    df_adh_validation_1 = adh_labeled[
        ['MOI', 'Segment Index', 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]', 'URI', 'Common Name', 'Genus', 'Species',
         'Foe', 'Threat Level']].fillna(0)
    df_adh_validation_1 = df_adh_validation_1.groupby(
        ['MOI', 'Common Name', 'Segment Index', 'URI', 'Genus', 'Species', 'Foe', 'Threat Level']).mean()


    df_adh_validation_2 = adh_labeled[
        ['MOI', 'Segment Index', 'cnt', 'URI', 'Common Name', 'Genus', 'Species', 'Foe', 'Threat Level']].fillna(0)
    df_adh_validation_2 = df_adh_validation_2.groupby(
        ['MOI', 'Common Name', 'Segment Index', 'URI', 'Genus', 'Species', 'Foe', 'Threat Level']).sum()

    # joining 2 dataframes
    df_adh_full = df_adh_validation_1.join(df_adh_validation_2)
    df_adh_full.rename({'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]': 'Avg Adherent Bacteria'}, axis=1, inplace=True)
    df_adh_full.reset_index(inplace=True)
    df_adh_full['index_col'] = range(len(df_adh_full))

    return df_adh_full

def fix_adh_cols(adh_df):
    adh_df_copy = adh_df.copy()
    cols = adh_df_copy.columns
    cols = [col.replace('Propidium Iodide 531,647', 'GFP 469,525') for col in cols] # change column names (TAMU switched dye from GFP to PI so switch it back here to be consistent)
    cols = [col.replace('Deconvolved_2', 'Deconvolved') for col in cols] # new change in the data
    adh_df_copy.columns = cols
    return adh_df_copy

def merge_multi_adh_data(dir):
    '''
    Load multiple adherence data from ADE output without processing them for ML
    '''
    strain_mapping_data = get_mapping_df()
    file_paths = [dir + '/' + file for file in os.listdir(dir)]
    dfs = []
    for file in file_paths:
        adh_df = pd.read_csv(file)
        adh_df = fix_adh_cols(adh_df)
        dfs.append(adh_df)
    # concat all the dfs
    concat_adh_df = pd.concat(dfs).reset_index(drop = True)
    concat_adh_df['Organism'] = concat_adh_df['Organism'].astype(str).str.lower()
    strain_mapping_data.Name = strain_mapping_data.Name.astype(str).str.lower()
    # merge adherence data with strain mapping file
    adh_labeled = pd.merge(concat_adh_df, strain_mapping_data[['Name', 'Common Name','Foe']], how = 'inner',
                           left_on = 'Organism', right_on = 'Name')
    
    adh_labeled['index_col'] = range(len(adh_labeled))
    return adh_labeled

def get_nist_baseline_scores():
    path = r'C:\Users\bbche\Documents\GitRepos\sttr_fof\data\output\baseline_model2'
    nist_baseline_data = get_test_data('baseline', 0.8, 'random', path)
    nist_path_score = nist_baseline_data.groupby(['Common Name'])['Foe_prob_predictions'].mean()
    return nist_path_score


def process_a549(adherence_dirs, add_baseline_score = True):
    '''
    Hardcoded function to import and process adherence assays of all IVV strains with A549
    '''
    strain_mapping_data = get_mapping_df()
    a549_dfs = []
    cols_to_keep = ['MOI', 'Organism', 'Segment Index', '#', 
                    'Size', 'Area', 'Mean[Deconvolved[Tsf[DAPI 377,447]]]', 'Area[Deconvolved[Tsf[GFP 469,525]]]', 
                    'Mean[Deconvolved[Tsf[GFP 469,525]]]', 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]', 
                    'Area_2[Deconvolved[Tsf[GFP 469,525]]]', 'Mean_2[Deconvolved[Tsf[GFP 469,525]]]', 
                    'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]']
    for path in adherence_dirs:
        a549_df = process_adherence_dir(path)
        a549_df = a549_df[cols_to_keep]
        a549_dfs.append(a549_df)
    a549_dfs = pd.concat(a549_dfs).reset_index(drop = True)
    
    a549_features1 = get_agg_host_n_bac(a549_dfs, 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]')
    a549_features2 = get_agg_host_n_bac(a549_dfs, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]')
    a549_merged = pd.merge(a549_features1, a549_features2, left_on = ['Common Name','Segment Index'], right_on = ['Common Name','Segment Index'], suffixes = ['_1','_2'])
    
    foe_dict = pd.Series(strain_mapping_data.Foe.values,index=strain_mapping_data['Common Name']).to_dict() # map foe label from strain name
    a549_merged['Foe'] = a549_merged['Common Name'].map(foe_dict)
    a549_merged.Foe = a549_merged.Foe.astype(int)
    
    feature_cols = [ '#_1', 'bac_cnt_mean_1', 'bac_cnt_max_1', 'bac_cnt_min_1', 'Area_mean_1', 
                    '#_2', 'bac_cnt_mean_2', 'bac_cnt_max_2', 'bac_cnt_min_2', 'Area_mean_2']
    scaler = MinMaxScaler()
    for col in feature_cols:
        a549_merged[f'scaled_{col}'] = scaler.fit_transform(a549_merged[col].values.reshape(len(a549_merged), 1))    
#     # add org_index for entropy loss to keep track of strain label
    le = LabelEncoder()
    a549_merged['org_index'] = le.fit_transform(a549_merged['Common Name'])
    
    # add the baseline score as a feature for baseline loss
    if add_baseline_score == True:
        baseline_foe_proba = get_nist_baseline_scores().reset_index()
        a549_merged = pd.merge(a549_merged, baseline_foe_proba, left_on = 'Common Name', right_on = 'Common Name')
    
    scaled_features = ['scaled_' + col for col in feature_cols]
    return a549_merged, scaled_features 


def process_adherence_dir(dir):
    file_paths = [dir + '/' + file for file in os.listdir(dir)]
    dfs = []    
    for file in file_paths:
        adh_df = pd.read_csv(file)
        cols = adh_df.columns
        cols = [col.replace('Propidium Iodide 531,647', 'GFP 469,525') for col in cols] # change column names (TAMU switched dye from GFP to PI so switch it back here to be consistent)
        cols = [col.replace('Deconvolved_2', 'Deconvolved') for col in cols] # new change in the data
        adh_df.columns = cols
        adh_df['Organism'] = adh_df['Organism'].astype(str)
        dfs.append(adh_df)
    return pd.concat(dfs)

def get_agg_host_n_bac(adh_df, spotcount_col):
    strain_mapping_data = get_mapping_df()
    if 'Common Name' not in adh_df.columns:
        adh_df.Organism = adh_df.Organism.astype(str).str.lower()
        adh_df = pd.merge(adh_df, strain_mapping_data, left_on = 'Organism', right_on = 'Name')
    host_per_pic = adh_df.groupby(['Common Name','Segment Index'])['#'].max().reset_index()
    bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].mean().reset_index().rename(columns = {spotcount_col:'bac_cnt_mean'})
    max_bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].max().reset_index().rename(columns = {spotcount_col:'bac_cnt_max'})
    min_bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].min().reset_index().rename(columns = {spotcount_col:'bac_cnt_min'})
    avg_area_per_pic = adh_df.groupby(['Common Name','Segment Index'])['Area'].mean().reset_index().rename(columns = {'Area':'Area_mean'})
    to_merge = [host_per_pic, bac_per_pic, max_bac_per_pic, min_bac_per_pic, avg_area_per_pic]
    merged_df = reduce(lambda left,right: pd.merge(left,right,on=['Common Name','Segment Index']), to_merge)
    return merged_df

def get_test_data(assay_name, percent, model_name, th_path):
    '''
    A function to query leaderboard and get test data in the test harness output directory
    Return test data in a df
    '''
    percent_df = pd.DataFrame(columns = ['% of Total Samples In Train', 'Balanced Accuracy'])
    df_leaderboard = query_leaderboard(query={'Description':assay_name+'__'+str(percent),
                                             'Model Name':model_name,
                                             'Column Predicted':'Foe'},
                                       th_output_location=th_path, classification=True)

    run_ids = df_leaderboard['Run ID'].tolist()
    # print(assay_name,percent,len(run_ids))
    test_dfs = []
    for run_id in run_ids:
        # for loop to read test data from 4 runs
        test_df = pd.read_csv(list(get_result_csv_paths([run_id], th_output_location=th_path,
                                                        file_type='testing_data').keys())[0])
        test_dfs.append(test_df)

    test_dfs = pd.concat(test_dfs)
    return test_dfs

def get_nist_baseline_scores():
    path = r'C:\Users\bbche\Documents\GitRepos\sttr_fof\data\output\baseline_model2'
    nist_baseline_data = get_test_data('baseline', 0.8, 'random', path)
    nist_path_score = nist_baseline_data.groupby(['Common Name'])['Foe_prob_predictions'].mean()
    return nist_path_score
def process_huvec_a549():
    '''
    Hardcoded function to import and process adherence assays of original 27 IVV strains with A549 and Huvec host cells
    '''
    strain_mapping_data = get_mapping_df()
    huvec_df = process_adherence_dir(prefix+'/Shared drives/STTR_Netrias/Project Stage/Data/Jing/Processed/old_ivv_huvec')
    a549_df = process_adherence_dir(prefix+'/Shared drives/STTR_Netrias/Project Stage/Data/Jing/Processed/old_ivv_A549')
    
    a549_features1 = get_agg_host_n_bac(a549_df, 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]')
    # drop '#' and 'Area_mean' from the huvec_features2 since these 2 columns are already present in huvec_features1
#     a549_features2 = get_agg_host_n_bac(a549_df, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]').drop(columns = ['#', 'Area_mean']) 
# apparently duplicating the # and Area_mean gave better results even though #_1 and #_2 are the same values
    a549_features2 = get_agg_host_n_bac(a549_df, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]')
    a549_merged = pd.merge(a549_features1, a549_features2, left_on = ['Common Name','Segment Index'], right_on = ['Common Name','Segment Index'], suffixes = ['_1','_2'])
    
    huvec_features1 = get_agg_host_n_bac(huvec_df, 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]')
#     huvec_features2 = get_agg_host_n_bac(huvec_df, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]').drop(columns = ['#', 'Area_mean'])
    huvec_features2 = get_agg_host_n_bac(huvec_df, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]')
    huvec_merged = pd.merge(huvec_features1, huvec_features2, left_on = ['Common Name','Segment Index'], right_on = ['Common Name','Segment Index'], suffixes = ['_1','_2'])
    
    joined_all = pd.merge(a549_merged, huvec_merged, left_on = 'Common Name', right_on = 'Common Name', suffixes = ['_a549', '_huvec'])
    
    foe_dict = pd.Series(strain_mapping_data.Foe.values,index=strain_mapping_data['Common Name']).to_dict() # map foe label from strain name
    joined_all['Foe'] = joined_all['Common Name'].map(foe_dict)
    joined_all.Foe = joined_all.Foe.astype(int)
#     joined_all['index_col'] = range(len(joined_all))
    feature_cols = [ '#_1_a549', 'bac_cnt_mean_1_a549', 'bac_cnt_max_1_a549', 'bac_cnt_min_1_a549', 'Area_mean_1_a549', 
                    '#_2_a549', 'bac_cnt_mean_2_a549', 'bac_cnt_max_2_a549', 'bac_cnt_min_2_a549', 'Area_mean_2_a549', 
                    '#_1_huvec', 'bac_cnt_mean_1_huvec', 'bac_cnt_max_1_huvec', 'bac_cnt_min_1_huvec', 'Area_mean_1_huvec', 
                    '#_2_huvec', 'bac_cnt_mean_2_huvec', 'bac_cnt_max_2_huvec', 'bac_cnt_min_2_huvec', 'Area_mean_2_huvec']
    scaler = MinMaxScaler()
    for col in feature_cols:
        joined_all[f'scaled_{col}'] = scaler.fit_transform(joined_all[col].values.reshape(len(joined_all), 1))
    
    # add org_index for entropy loss to keep track of strain label
    le = LabelEncoder()
    joined_all['org_index'] = le.fit_transform(joined_all['Common Name'])
    
    # add the baseline score as a feature for baseline loss
    baseline_foe_proba = get_nist_baseline_scores().reset_index()
    joined_all = pd.merge(joined_all, baseline_foe_proba, left_on = 'Common Name', right_on = 'Common Name')
    
    scaled_features = ['scaled_' + col for col in feature_cols]
    cols_to_keep = scaled_features + ['Common Name', 'Foe', 'org_index', 'Foe_prob_predictions']
    return joined_all[cols_to_keep], scaled_features 


def amr_data(amr_file, antibiotics = []):
    '''
    amr_file: The amr data, will change this to a directory if file numbers increase
    MIC: Minimal inhibitory concentration as the model feature. This is the default setting.
    return: Dataframe ready for analysis for the adherence model
    '''
    strain_mapping_df = get_mapping_df()
    amr_df = pd.read_excel(amr_file)
    amr_df['Strain ID'] = amr_df['Strain ID'].apply(str).apply(str.lower)

    amr_labeled = pd.merge(amr_df, strain_mapping_df, how='inner', right_on='Name', left_on='Strain ID')
    antibiotic_cols = [ab + suf for ab in antibiotics for suf in ['_1', '_2', '_3']]
    antibiotic_proc_cols = [ab + '_Proc' for ab in antibiotics]
    # We don't analyze SIR data anymore so only DD is used
    id_cols = list(set(amr_labeled.columns) - set(antibiotic_cols))
    df_amr_ivv_mic = pd.melt(amr_labeled, id_vars=id_cols, value_vars=antibiotic_cols, var_name='antibiotic',
                             value_name='DD')
    df_amr_ivv_mic['antibiotic'] = df_amr_ivv_mic['antibiotic'].str[:-2]
    df_amr_ivv_mic['index_col'] = range(len(df_amr_ivv_mic))
    df_amr_ivv_mic['Foe'] = df_amr_ivv_mic['Foe'].astype(int)
    return df_amr_ivv_mic

def add_new_features(amr_df):
    amr_df_copy = amr_df.copy()
    amr_df_copy['DD^2'] = amr_df_copy['DD'].apply(lambda x:x**2)
    
    amr_df_copy['DD^2_std'] = 0
    for org in amr_df_copy['Common Name'].unique():
        for antibiotic in amr_df_copy['antibiotic'].unique():
            subset = amr_df_copy[(amr_df_copy['Common Name'] == org) & (amr_df_copy['antibiotic'] == antibiotic)]
            subset_std = subset['DD^2'].std()
            amr_df_copy.loc[((amr_df_copy['Common Name'] == org) & (amr_df_copy['antibiotic'] == antibiotic)), 'DD^2_std'] = subset_std
            
    return amr_df_copy

def reporter_data_no_FlowCal(dir, files_to_use, strain_mapping_data):
    '''
    Load reporter data from a directory and transform it into the dataframe needed for the model
    dir: A directory where reporter data are
    strain_mapping_data: Excel file for mapping strain names
    return: Dataframe ready for analysis for the reporter model
    '''
    file_paths = [dir + '/' + file for file in files_to_use]
    dfs = []
    for file in file_paths:
        reporter_df = pd.read_csv(file, low_memory = False)
        # only save the columns that will be useful for training
        reporter_df = reporter_df[['Organism', 'FSC-A', 'FSC-H', 'FSC-W',
                                         'SSC-A', 'SSC-H', 'SSC-W', 'FITC-A', 'PE-Texas Red-A']]
        # remove 'No Stain', 'Dead-No Dye' and 'Dead-R' samples
        reporter_df = reporter_df[~reporter_df['Organism'].isin(['No Stain', 'Dead-No Dye', 'Dead-R'])]
        dfs.append(reporter_df)


    # randomly take 2000 cells from each organism to reduce sample size
    new_dfs = []
    for df in dfs:
        for org in df['Organism'].unique():
            if not pd.isna(org):
                df_sub = df[df['Organism'] == org]
                if len(df_sub) > 0:
                    df_sub = df_sub.sample(2000)
                    new_dfs.append(df_sub)

    # reduce function to merge all the dfs
    merged_reporter = pd.concat(new_dfs)
    # Apply log10 to each channel
    channels = ['FSC-A', 'FSC-H', 'FSC-W','SSC-A', 'SSC-H', 'SSC-W', 'FITC-A', 'PE-Texas Red-A']
    log_channels = ['log_' + channel for channel in channels]
    log_reporter = pd.DataFrame(columns = log_channels)
    for i, log_channel in enumerate(log_channels):
        log_reporter[log_channel] = np.log10(merged_reporter[channels[i]])

    log_reporter['Organism'] = merged_reporter['Organism']
    # Hard gate the cells
    log_reporter = log_reporter[(log_reporter['log_FSC-A'] > 4.45) & (log_reporter['log_SSC-A'] > 3.1)]
    # remove +/-infinity, nan values
    log_reporter = log_reporter[~log_reporter.isin([np.nan, np.inf, -np.inf]).any(1)]
    # mapping strain names
    log_reporter['Organism'] = log_reporter['Organism'].str.lower()
    log_reporter = pd.merge(log_reporter, strain_mapping_data, how = 'inner', left_on = 'Organism', right_on = 'Name')
    log_reporter = log_reporter[~log_reporter['Genus'].isin(['no genus'])]

    log_reporter['index_col'] = range(len(log_reporter))
    log_reporter['Foe'] = log_reporter['Foe'].astype(int)
    return log_reporter



def toxicity_data(dir, strain_mapping_data):
    '''
    Load toxicity data from a directory and transform it into the dataframe needed for the model
    dir: A directory where toxicity data are
    strain_mapping_data: Excel file for mapping strain names
    return: Dataframe ready for analysis for the toxicity model
    '''
    file_paths = [dir + '/' + file for file in os.listdir(dir)]
    dfs = []
    for file in file_paths:
        if 'csv' in file:
            toxicity_df = pd.read_csv(file)
            dfs.append(toxicity_df)

    # reduce function to merge all the dfs
    merged_tox_df = reduce(lambda left, right: pd.merge(left, right, how = 'outer'), dfs)
    merged_tox_df['Organism'] = merged_tox_df['Organism'].str.lower()
    merged_tox_df['Organism'] = merged_tox_df['Organism'].str.replace('.0', '', regex=False)
    merged_tox_df['Organism'] = merged_tox_df['Organism'].str.replace(u'\xa0', u'')
    # merge adherence data with strain mapping file
    tox_labeled = pd.merge(merged_tox_df, strain_mapping_data, how = 'inner',
                           left_on = 'Organism', right_on = 'Name')

    # Assign time windows
    tox_labeled['Time_Window'] = 'window'

    tox_labeled['Time_Window'][tox_labeled.Time.between(0, 3)] = "0-4 hours"
    tox_labeled['Time_Window'][tox_labeled.Time.between(4, 7)] = "4-8 hours"
    tox_labeled['Time_Window'][tox_labeled.Time.between(8, 11)] = "8-12 hours"
    tox_labeled['Time_Window'][tox_labeled.Time.between(12, 15)] = "12-16 hours"
    tox_labeled['Time_Window'][tox_labeled.Time.between(16, 18)] = "16-18 hours"
    # remove non-bacterial samples
    tox_labeled = tox_labeled[~tox_labeled['Genus'].isin(['no genus'])]
    tox_labeled['index_col'] = range(len(tox_labeled))
    tox_labeled = tox_labeled[tox_labeled.Foe != 'no label']
    tox_labeled['Foe'] = tox_labeled['Foe'].astype(int)
    return tox_labeled



def toxicity_data2(dapi_path, hoechst_path, add_time = False):
    '''
    New method to load toxicity data to get raw dapi and hoechst 
    Sometimes TAMU uses PI for DAPI so use the PI_path for dapi_path
    '''
    strain_mapping_data = get_mapping_df()

    dapi_file = pd.read_csv(dapi_path)
    hoechst_file = pd.read_csv(hoechst_path)
    # The following 2 lines are only there for the specific df format used here. If anything changes, these need to change
    dapi_file = dapi_file.iloc[1:,2:]
    hoechst_file = hoechst_file.iloc[1:,2:]
    col_names = dapi_file.columns.tolist()
    # Time variable should be present in the dataframe but in this case it was missing so I manually added it back
    if add_time == True:
        dapi_file['Time'] = range(19)
        hoechst_file['Time'] = range(19)
    dapi_df = pd.melt(dapi_file, id_vars = 'Time', value_vars = col_names, value_name='DAPI')
    hoechst_df = pd.melt(hoechst_file, id_vars = 'Time', value_vars = col_names, value_name='Hoechst')
    
    joined = pd.merge(dapi_df, hoechst_df, left_on = ['Time','variable'], right_on = ['Time','variable'])
    joined['variable'] = joined['variable'].str.replace('\.1','')
    joined['variable'] = joined['variable'].str.replace('\.2','')
    joined['variable'] = joined['variable'].str.replace('\.3','')
    joined.variable = joined.variable.str.lower()
    joined_labeled = pd.merge(joined, strain_mapping_data, left_on = 'variable', right_on = 'Name')
    joined_labeled = joined_labeled[joined_labeled.Foe != 'no label']

    for col in ['Time', 'DAPI', 'Hoechst']:
        joined_labeled[col] = pd.to_numeric(joined_labeled[col])
        joined_labeled = joined_labeled.sort_values(['Foe', 'variable'])
    joined_labeled.Foe = joined_labeled.Foe.astype(int)
    return joined_labeled




## read_fcs, flow_scatter and gate_live_cells are functions for loading reporter assays with FlowCal
def read_fcs(path):
    '''
    path: path + filenames to fcs files
    return fcs files loaded with FlowCal
    '''
    file = FlowCal.io.FCSData(path)
    file = FlowCal.transform.to_rfi(file)
    return file

def flow_scatter(sample_name):
    '''
    sample_name: fcs data loaded with FlowCal
    return density plot with FSC-A and SSC-A as the axes
    '''
    FlowCal.plot.density2d(sample_name, channels = ['FSC-A','SSC-A'], mode = 'scatter')
    plt.xlim((10**4,10**6)) # the ideal x and y axes need some tuning (differ by machines)
    plt.ylim((10**3,10**6))
    plt.title(sample_name)
    plt.show()
    plt.close()


def gate_live_cells(samples, plot=False):
    '''
    samples: fcs files loaded with read_fcs
    return gated FlowCal data with live cells only
    '''
    # Gating cells by size
    live_cells = FlowCal.gate.ellipse(samples, channels=['FSC-A', 'SSC-A'],
                                      log=True, center=(5, 3.5),  # based on judgement
                                      a=1, b=0.5, theta=80 / 180. * np.pi)

    # Gating live cells by negative Texas Red
    Texas_neg = FlowCal.gate.high_low(live_cells, channels='PE-Texas Red-A',
                                      high=2 * 10 ** 3, low=0)

    if plot:
        plt.figure(figsize=(6, 8))
        plt.subplots_adjust(hspace=0.4)

        plt.subplot(211)
        flow_scatter(Texas_neg)
        plt.title('Gated live cells')

        plt.subplot(212)
        FlowCal.plot.hist1d(Texas_neg, channel='PE-Texas Red-A')
        plt.title('Texas Red signal from gated cells')
    return Texas_neg

def get_sample_name(filename):
    '''
    Parsing filenames from Google Drives including paths to get sample names as a column in the dataframe
    '''
    filename = filename.split('/')[-1]
    if '-NT' not in filename and '-T' not in filename:
        try:
            return filename.split('-')[0].split('_')[1]
        except:
            return filename.split('_')[1]
    else:
        sample_name = filename.split('_')[1].split('-')[:2]
        return '-'.join(sample_name)


def get_time_point(filename):
    '''
    Parsing filenames from Google Drives to get time points in the file names as a column in the dataframe
    :param filename
    :return: time points
    '''
    filename = filename.split('/')[-1]
    try:
        return filename.split('-')[1].split('_')[0]
    except:
        return filename.split('_')[2]
def reporter_data_w_FlowCal(dir_list, strain_mapping_data, time_point):
    '''
    Load reporter data from a directory and transform it into the dataframe needed for the model
    dir_list: A list of directories where the fcs data are
    strain_mapping_data: Excel file for mapping strain names
    time_point: if there is time point in the filename
    return: Dataframe ready for analysis for the reporter model
    '''
    fcs_data = []
    all_file_names = []
    for dir in dir_list:
        file_names = [dir + '/' + file for file in os.listdir(dir)]
        for file in file_names:
            if 'Dead' not in file and 'No Stain' not in file:
                fcs_data.append(read_fcs(file))
                all_file_names.append(file)

    # gate each loaded fcs file
    gated_data = []
    for data in fcs_data:
        gated_data.append(gate_live_cells(data))
    # unrolling each live cell into a row in a df
    live_dfs = []
    # channels to use
    channels = gated_data[0].channels[:-1]

    for index in range(len(all_file_names)):
        temp_df = pd.DataFrame(columns = channels) # initiate an empty df with each channel as a column
        for channel in channels: # loop through the channels
            temp_df[channel] = np.log10(gated_data[index][:,channel]) # apply log to each channel and adding to the df
            temp_df['file_name'] = all_file_names[index]
        if len(temp_df) > 2000:
            temp_df = temp_df.sample(n = 2000) # take 2000 cells from each sample
        live_dfs.append(temp_df)
    merged_df = pd.concat(live_dfs)
    merged_df['Organism'] = merged_df['file_name'].apply(lambda x: get_sample_name(x))

    # remove inf, -inf and nan in the data
    merged_df = merged_df[~merged_df.isin([np.inf, -np.inf])]
    merged_df = merged_df.dropna(axis=0)

    if time_point == True:
        merged_df['Time'] = merged_df['file_name'].apply(lambda x: get_time_point(x))
        merged_df = merged_df[~merged_df['Time'].isin(['30min', '1h', '2h', '3h', '4h'])] # only use time point 5h
    # map strain names
    merged_df['Organism'] = merged_df['Organism'].str.lower()
    merged_df = pd.merge(merged_df, strain_mapping_data, how='inner', right_on='Name', left_on='Organism')
    merged_df['index_col'] = range(len(merged_df))
    merged_df['Foe'] = merged_df['Foe'].astype(int)

    return merged_df


def balance_df(df):
    '''
    :param df: an assay df
    :return: balanced df with equal number of friends and foes
    '''
    friend_df = df[df.Foe == 0]
    foe_df = df[df.Foe == 1]
    if len(friend_df) > len(foe_df):
        friend_df = friend_df.sample(len(foe_df))
    else:
        foe_df = foe_df.sample(len(friend_df))
    joined = pd.concat([friend_df, foe_df])
    return joined


def process_reporter(reporter_df):
    '''
    :param df: an reporter df
    :return: scaled reporter assay for each channel and the scaled_features as a list
    '''
    # choose feature cols
    feature_cols = ['FSC-A', 'FSC-H', 'FSC-W', 'SSC-A',
                    'SSC-H', 'SSC-W', 'FITC-A', 'PE-Texas Red-A']
    # scale features
    scaler = MinMaxScaler()
    for col in feature_cols:
        reporter_df[f'scaled_{col}'] = scaler.fit_transform(reporter_df[col].values.reshape(len(reporter_df), 1))

    scaled_features = ['scaled_' + col for col in feature_cols]
    reporter_df['Foe'] = reporter_df['Foe'].astype(int)
    reporter_df = balance_df(reporter_df)
    # encode the 'Common Name' with numbers and feed it for neural network to keep track of the strain
    le = LabelEncoder()
    reporter_df['org_index'] = le.fit_transform(reporter_df['Common Name'])
    return reporter_df, scaled_features