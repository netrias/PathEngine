import pandas as pd
import os
from ..reference import fof_strain_mapping
from functools import reduce
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

strain_mapping_df = fof_strain_mapping.get_mapping_df()

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
    return pd.concat(dfs).reset_index(drop = True)

def get_agg_host_n_bac(adh_df, spotcount_col):
    if 'Common Name' not in adh_df.columns:
        adh_df.Organism = adh_df.Organism.astype(str).str.lower()
        adh_df = pd.merge(adh_df, strain_mapping_df, left_on = 'Organism', right_on = 'Name')
    host_per_pic = adh_df.groupby(['Common Name','Segment Index'])['#'].max().reset_index()
    bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].mean().reset_index().rename(columns = {spotcount_col:'bac_cnt_mean'})
    max_bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].max().reset_index().rename(columns = {spotcount_col:'bac_cnt_max'})
    min_bac_per_pic = adh_df.groupby(['Common Name','Segment Index'])[spotcount_col].min().reset_index().rename(columns = {spotcount_col:'bac_cnt_min'})
    avg_area_per_pic = adh_df.groupby(['Common Name','Segment Index'])['Area'].mean().reset_index().rename(columns = {'Area':'Area_mean'})
    to_merge = [host_per_pic, bac_per_pic, max_bac_per_pic, min_bac_per_pic, avg_area_per_pic]
    merged_df = reduce(lambda left,right: pd.merge(left,right,on=['Common Name','Segment Index']), to_merge)
    return merged_df


A549_COLS_TO_KEEP = ['MOI', 'Organism', 'Segment Index', '#',
                     'Size', 'Area', 'Mean[Deconvolved[Tsf[DAPI 377,447]]]', 'Area[Deconvolved[Tsf[GFP 469,525]]]',
                     'Mean[Deconvolved[Tsf[GFP 469,525]]]', 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]',
                     'Area_2[Deconvolved[Tsf[GFP 469,525]]]', 'Mean_2[Deconvolved[Tsf[GFP 469,525]]]',
                     'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]']


def process_a549_files(adherence_dirs):
    '''
    Provide directories and process adherence assays of all strains done with the A549 host line
    '''
    a549_dfs = []
    for path in adherence_dirs:
        a549_df = process_adherence_dir(path)
        a549_dfs.append(a549_df[A549_COLS_TO_KEEP])
    a549_dfs = pd.concat(a549_dfs).reset_index(drop=True)
    return process_a549_df(a549_dfs)

def process_a549_df(a549_dfs):
    a549_dfs = a549_dfs[A549_COLS_TO_KEEP]

    a549_features1 = get_agg_host_n_bac(a549_dfs, 'SpotCount[Deconvolved[Tsf[GFP 469,525]]]')
    a549_features2 = get_agg_host_n_bac(a549_dfs, 'SpotCount_2[Deconvolved[Tsf[GFP 469,525]]]')
    a549_merged = pd.merge(a549_features1, a549_features2, left_on=['Common Name', 'Segment Index'],
                           right_on=['Common Name', 'Segment Index'], suffixes=['_1', '_2'])

    foe_dict = pd.Series(strain_mapping_df.Foe.values,
                         index=strain_mapping_df['Common Name']).to_dict()  # map foe label from strain name
    a549_merged['Foe'] = a549_merged['Common Name'].map(foe_dict)

    feature_cols = ['#_1', 'bac_cnt_mean_1', 'bac_cnt_max_1', 'bac_cnt_min_1', 'Area_mean_1',
                    '#_2', 'bac_cnt_mean_2', 'bac_cnt_max_2', 'bac_cnt_min_2', 'Area_mean_2']
    scaler = MinMaxScaler()
    for col in feature_cols:
        new_col_name = f'{col}_a549'
        a549_merged.rename(columns = {col:new_col_name}, inplace = True)
        a549_merged[f'scaled_{new_col_name}'] = scaler.fit_transform(a549_merged[new_col_name].values.reshape(len(a549_merged), 1))
    # add org_index for entropy loss to keep track of strain label
    le = LabelEncoder()
    a549_merged['org_index'] = le.fit_transform(a549_merged['Common Name'])

    scaled_features = [f'scaled_{col}_a549' for col in feature_cols]
    a549_merged.Foe = a549_merged.Foe.astype(int)
    a549_merged['index_col'] = range(len(a549_merged)) # for test-harness

    return a549_merged, scaled_features