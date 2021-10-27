import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from harness.utils.parsing_results import *
from harness.test_harness_class import TestHarness
from sklearn.metrics import balanced_accuracy_score
from ..preprocessing.reference import fof_strain_mapping
from functools import reduce

strain_mapping_data = fof_strain_mapping.get_mapping_df()
foe_dict = pd.Series(strain_mapping_data['Foe'].values, index = strain_mapping_data['Common Name']).to_dict()

def get_strain_mapping_short():
    strain_mapping_short = strain_mapping_data[['Common Name','Foe']]
    strain_mapping_short = strain_mapping_short[strain_mapping_short!='no label']
    strain_mapping_short.dropna(inplace = True)
    strain_mapping_short['Foe'] = strain_mapping_short['Foe'].astype(int)
    return strain_mapping_short

def get_common_to_genus_species():
    strain_mapping_copy = strain_mapping_data.copy()
    strain_mapping_copy['genus_species'] = strain_mapping_copy['Genus'].str.capitalize() + ' '+ strain_mapping_copy.Species
    name_dict = pd.Series(strain_mapping_copy['genus_species'].values, index = strain_mapping_copy['Common Name']).to_dict()
    return name_dict


def stage2_balanced_accuracy(assay, model_name, th_path):
    '''
    assay: assay name, must be the same as what's used in test-harness
    model_name: best performing model to query 
    '''
    strain_mapping_short = get_strain_mapping_short()
    percents=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    assay_acc = []
    for percent in percents:
        percent_df = pd.DataFrame(columns = ['% of Total Samples In Train', 'Balanced Accuracy'])
        df_leaderboard = query_leaderboard(query={'Description':assay+'__'+str(percent),
                                                 'Model Name':model_name,
                                                 'Column Predicted':'Foe'}, 
                                           th_output_location=th_path, classification=True)

        run_ids = df_leaderboard['Run ID'].tolist()
        print(assay,percent,len(run_ids))
        accuracy_per_run = []
        for run_id in run_ids:
            # for loop to read test data from 4 runs
            test_df = pd.read_csv(list(get_result_csv_paths([run_id], th_output_location=th_path).keys())[0])
            # getting the mean of Foe_prob_predictions
            avg_prob_df = test_df.groupby(['Common Name'])['Foe_predictions'].mean().reset_index()
            avg_prob_df['agg_foe_prediction'] = 0
            # based on the aggregated Foe_prob_predictions to make a call if a strain is a friend or foe
            avg_prob_df.loc[avg_prob_df['Foe_predictions'].between(0.45, 0.55), 'agg_foe_prediction'] = 'NS'
            avg_prob_df.loc[avg_prob_df['Foe_predictions'] > 0.55, 'agg_foe_prediction'] = 1    
# use the line below the comment out 2 previous line to change the function to no threshold
#             avg_prob_df.loc[avg_prob_df['Foe_prob_predictions'] > 0.5, 'agg_foe_prediction'] = 1    
            
            # merge with strain_mapping_short to get the actual Foe label
            label_avg_prob_df = avg_prob_df.merge(strain_mapping_short, left_on = 'Common Name', right_on = 'Common Name')
            label_avg_prob_df = label_avg_prob_df[label_avg_prob_df['agg_foe_prediction']!= 'NS']
            
            label_avg_prob_df['agg_foe_prediction'] = label_avg_prob_df['agg_foe_prediction'].astype(int)
            # get the balanced score 
            score = balanced_accuracy_score(label_avg_prob_df['Foe'],label_avg_prob_df['agg_foe_prediction'])
            accuracy_per_run.append(100*score)
        # A df to store the accuracy from each percent    
        percent_df['Balanced Accuracy'] = accuracy_per_run
        percent_df['% of Total Samples In Train'] = 100*percent
        assay_acc.append(percent_df)
    # Concatenate all runs on 7 percents
    assay_acc = pd.concat(assay_acc)
    assay_acc['% of Total Samples In Train'] = assay_acc['% of Total Samples In Train'].astype(int)
    return assay_acc

def plot_stage2_aggregated_accuracy(aggregated_score_df, assay_name, output_dir = None):
    ax = sns.boxplot(data = aggregated_score_df, x = '% of Total Samples In Train', y = 'Balanced Accuracy')  
    ax.set(ylim = (0,100))
    plt.axhline(50, color = 'r')
    plt.title(assay_name)
    if output_dir != None:
        plt.savefig(f'{output_dir}/stage2_{assay_name}.png', bbox_inches = 'tight')
        print(f'Figure has been saved.')
    plt.show()
    plt.close()

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


def get_stage3_performance1(assay_test_results, assay_names, threshold = 0, strain_mapping_data = strain_mapping_data):
    '''
    A way to measure stage 3 performance by sampling each assay with the smallest sample size of all assays
    Then apply weighted average on all the data points from all assasy to get the foe_proba to determine if a strain is a friend or a foe
    The downside of this method is if an assay is much bigger than the rest, it is under-represented when it's subsampled
    '''
    assay_tests_w_threshold = []
    for test_df in assay_test_results:
        test_df = test_df[~test_df['Foe_prob_predictions'].between(0.5-threshold, 0.5+threshold)]
        assay_tests_w_threshold.append(test_df)
    min_sample = min([len(test_df) for test_df in assay_tests_w_threshold])
    joined_test_df = []
    acc_by_assay = []
    for i, test_df in enumerate(assay_tests_w_threshold):
        test_df_subset = test_df.sample(min_sample)
        joined_test_df.append(test_df_subset)

    joined_test_df = pd.concat(joined_test_df).reset_index(drop = True)
    joined_test_df['Common Name'] = joined_test_df['Common Name'].str.replace('Salmonella enterica ', 'Salmonella enterica')
    joined_test_df = joined_test_df[joined_test_df['Common Name'].str.contains('NIST')] # only keep NIST strains
    # determine stage 3 accuracy (overall accuracy)
    stage3_mean = joined_test_df.groupby(['Common Name'])['Foe_prob_predictions'].mean().reset_index()
    stage3_mean['Foe_pred'] = stage3_mean['Foe_prob_predictions'].apply(lambda x: 0 if x < 0.5 else 1) 
    stage3_mean['Foe'] = stage3_mean['Common Name'].map(foe_dict).astype(int)
    stage3_accuracy = 100*accuracy_score(stage3_mean.Foe, stage3_mean.Foe_pred)
    stage3_precision = 100*precision_score(stage3_mean.Foe, stage3_mean.Foe_pred)
    stage3_recall = 100*recall_score(stage3_mean.Foe, stage3_mean.Foe_pred)
    stage3_f1 = 100*f1_score(stage3_mean.Foe, stage3_mean.Foe_pred)
    return stage3_accuracy, stage3_precision, stage3_recall, stage3_f1
    
def get_different_stage3_combo1(adh_result, tox_result, reporter_result, amr_result, threshold = 0.05):
    all_4 = []
    adh_tox_amr = []
    tox_rep_amr = []
    tox_amr = []
    adh_tox = []
    adh_rep = []
    adh_amr = []
    df = pd.DataFrame(columns = {'Assays', 'Accuracy', 'Precision', 'Recall', 'F1'})
    for i in range(8):
        acc, prec, recall, f1 = get_stage3_performance1([adh_result, tox_result, reporter_result, amr_result], 
                                                             ['adh', 'tox', 'immune', 'ar'], threshold)
        
        df.loc[i] = ['all 4', acc, prec, recall, f1]
        
        acc, prec, recall, f1 = get_stage3_performance1([adh_result, tox_result, amr_result], 
                                                             ['adh', 'tox', 'ar'], threshold)
        
        df.loc[i+8] = ['adh_tox_ar', acc, prec, recall, f1]   
        
        acc, prec, recall, f1 = get_stage3_performance1([tox_result, amr_result], 
                                                        ['tox', 'ar'], threshold)
        
        df.loc[i+16] = ['tox_ar', acc, prec, recall, f1]   
        
        acc, prec, recall, f1 = get_stage3_performance1([adh_result, tox_result], 
                                                        ['adh', 'tox'], threshold)
        
        df.loc[i+24] = ['adh_tox', acc, prec, recall, f1]  
        acc, prec, recall, f1 = get_stage3_performance1([adh_result, amr_result], 
                                                        ['adh', 'ar'], threshold)
        
        df.loc[i+32] = ['adh_ar', acc, prec, recall, f1]  
        
    df.columns = ['Assays', 'Accuracy', 'Precision', 'Recall', 'F1']
    return df


def get_th_foe_proba(assay_name, percent, model_name, th_path, assay_data_to_predict,
                      feature_cols, feature_cols_to_normalize, sparse_cols_to_use = None,
                      training_data = None, testing_data = None):
    df_leaderboard = query_leaderboard(query={'Description':assay_name+'__'+str(percent),
                                             'Model Name':model_name,
                                             'Column Predicted':'Foe'},
                                       th_output_location=th_path, classification=True)

    run_ids = df_leaderboard['Run ID'].tolist()
    try: 
        prediction_data = get_predict_data(assay_name, percent, model_name, th_path)
    except:
        th = TestHarness(output_location = th_path)
        for run in run_ids:
            th.predict_only(run, assay_data_to_predict, index_cols = ['index_col', 'Common Name'], target_col = 'Foe', feature_cols_to_use = feature_cols,
                            feature_cols_to_normalize = feature_cols_to_normalize, sparse_cols_to_use = sparse_cols_to_use, 
                            training_data = training_data, testing_data = testing_data)
        prediction_data = get_predict_data(assay_name, percent, model_name, th_path)
    return prediction_data

def get_stage3_prediction(assay_test_results, assay_names, threshold = 0, strain_mapping_data = strain_mapping_data):
    '''
    A way to measure stage 3 performance by sampling each assay with the smallest sample size of all assays
    Then apply weighted average on all the data points from all assasy to get the foe_proba to determine if a strain is a friend or a foe
    The downside of this method is if an assay is much bigger than the rest, it is under-represented when it's subsampled
    '''
    assay_tests_w_threshold = []
    for test_df in assay_test_results:
        test_df = test_df[~test_df['Foe_prob_predictions'].between(0.5-threshold, 0.5+threshold)]
        assay_tests_w_threshold.append(test_df)
    min_sample = min([len(test_df) for test_df in assay_tests_w_threshold])
    joined_test_df = []
    acc_by_assay = []
    for i, test_df in enumerate(assay_tests_w_threshold):
        test_df_subset = test_df.sample(min_sample)
        joined_test_df.append(test_df_subset)

    joined_test_df = pd.concat(joined_test_df).reset_index(drop = True)
    joined_test_df['Common Name'] = joined_test_df['Common Name'].str.replace('Salmonella enterica ', 'Salmonella enterica')
    # determine stage 3 accuracy (overall accuracy)
    stage3_mean = joined_test_df.groupby(['Common Name'])['Foe_prob_predictions'].mean().reset_index()
    stage3_mean['Foe_pred'] = stage3_mean['Foe_prob_predictions'].apply(lambda x: 0 if x < 0.5 else 1) 
    return stage3_mean

def repeat_stage3_aggregation(adh_data, tox_data, rep_data, amr_data):
    all_4_acc = []
    for i in range(8):
        acc = get_stage3_prediction([adh_data, tox_data, rep_data, amr_data], ['adh', 'tox', 'reporter', 'amr'],0.07)
        all_4_acc.append(acc)
    all_4_acc = pd.concat(all_4_acc).reset_index(drop = True)
    agg_all_4 = all_4_acc.groupby(['Common Name'])['Foe_prob_predictions'].mean().reset_index()
    agg_all_4['Foe_prediction'] = agg_all_4['Foe_prob_predictions'].apply(lambda x: 'friend' if x < 0.5 else 'foe') 
    agg_all_4.rename(columns = {'Common Name':'Strain', 'Foe_prob_predictions':'Foe probability'}, inplace = True)
    return agg_all_4

def get_stage3_mean(assay_test_results, threshold = 0, strain_mapping_data = strain_mapping_data):
    '''
    A way to measure stage 3 performance by sampling each assay with the smallest sample size of all assays
    Then apply weighted average on all the data points from all assasy to get the foe_proba to determine if a strain is a friend or a foe
    The downside of this method is if an assay is much bigger than the rest, it is under-represented when it's subsampled
    '''
    assay_tests_w_threshold = []
    for test_df in assay_test_results:
        test_df = test_df[~test_df['Foe_prob_predictions'].between(0.5-threshold, 0.5+threshold)]
        assay_tests_w_threshold.append(test_df)
    min_sample = min([len(test_df) for test_df in assay_tests_w_threshold])
    joined_test_df = []
    acc_by_assay = []
    for i, test_df in enumerate(assay_tests_w_threshold):
        test_df_subset = test_df.sample(min_sample)
        joined_test_df.append(test_df_subset)

    joined_test_df = pd.concat(joined_test_df).reset_index(drop = True)
    joined_test_df['Common Name'] = joined_test_df['Common Name'].str.replace('Salmonella enterica ', 'Salmonella enterica')
    joined_test_df = joined_test_df[joined_test_df['Common Name'].str.contains('NIST')]
    # determine stage 3 accuracy (overall accuracy)
    stage3_mean = joined_test_df.groupby(['Common Name'])['Foe_prob_predictions'].mean().reset_index()
    stage3_mean['Foe_pred'] = stage3_mean['Foe_prob_predictions'].apply(lambda x: 0 if x < 0.5 else 1) 
    stage3_mean['Foe'] = stage3_mean['Common Name'].map(foe_dict).astype(int)
    return stage3_mean

def stage3_by_strain(best_assay_data_list, thresh):
    '''Get stage 3 (final prediction for each strain by aggregating across assays) predictions'''
    stage3_all = []
    for i in range(8):
        stage3_mean = get_stage3_mean(best_assay_data_list,0.02)
        stage3_all.append(stage3_mean)
    stage3_all = pd.concat(stage3_all)   
    stage3_agg = stage3_all.groupby('Common Name')['Foe_prob_predictions'].mean().reset_index()
    stage3_agg['Foe'] = stage3_agg['Common Name'].map(foe_dict)
    stage3_agg['Foe'] = stage3_agg['Foe'].astype(int)
    stage3_agg['Foe_pred'] = stage3_agg['Foe_prob_predictions'].apply(lambda x: 0 if x < 0.5 else 1)
    stage3_agg['Foe_pred'] = stage3_agg['Foe_pred'].astype(int)
    stage3_agg.index = stage3_agg['Common Name']
    acc = balanced_accuracy_score(stage3_agg.Foe, stage3_agg.Foe_pred)
    print('Accuracy', round(acc, 3))
    return stage3_agg

def plot_strain_level_heatmap(assay_data_list, assay_name_list, stage3_mean):
    '''A function to plot all the predictions per assay and in the ensemble prediction'''
    name_dict = get_common_to_genus_species()
    assay_mean_list = []
    for assay in assay_data_list:
        new_assay = assay[~assay.Foe_prob_predictions.between(0.45,0.55)]
        assay_mean = new_assay.groupby('Common Name')['Foe_prob_predictions'].mean()
        assay_mean_list.append(assay_mean)
    joined_df = reduce(lambda left, right: pd.merge(left, right, left_index = True, right_index = True), assay_mean_list)
    joined_df = pd.merge(joined_df, stage3_mean, left_index = True, right_index = True)
    
    joined_df['Foe'] = joined_df.index.map(foe_dict)
    joined_df = joined_df[joined_df.index.str.contains('NIST')]
    joined_df['index_col'] = joined_df.index.map(name_dict)
    joined_df = joined_df.sort_values(['Foe','index_col'])
    joined_df = joined_df.drop(columns = ['index_col'])
    xticklabels = assay_name_list + ['Ensemble probabilities','Ensemble predictions','Pathogenicity label']
    joined_df.columns = xticklabels
    yticklabels = joined_df.index.map(name_dict)
#     joined_df_t = joined_df.T
    sns.set(rc={'figure.figsize':(6,15)},font_scale=1.6)
    sns.heatmap(joined_df, center = 0.5, cmap = 'coolwarm',
                xticklabels = xticklabels, yticklabels = yticklabels)
#                 xticklabels=joined_df.columns.tolist(), yticklabels=yticklabels)
    plt.xticks(rotation=40, ha='right')
    plt.xlabel('')
    plt.ylabel('')
    return joined_df