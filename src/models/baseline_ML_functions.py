import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas as pd
from harness.test_harness_class import TestHarness
from sklearn.model_selection import train_test_split
from harness.th_model_instances.hamed_models.random_forest_classification import random_forest_classification
from harness.th_model_instances.hamed_models.rocklin_models import logistic_classifier
from harness.th_model_instances.perovskite_models.xgboost import gradient_boosted_tree
from harness.th_model_instances.hamed_models.navie_bayes import gaussian_naive_bayes_classification
from harness.utils.parsing_results import *
from src.preprocessing.load_data import get_mapping_df
import seaborn as sns
from functools import reduce
from numpy import trapz
from sys import platform 
prefix = 'G:' if platform == 'win32' else '/Volumes/GoogleDrive'

strain_mapping_data = get_mapping_df()
strain_label = strain_mapping_data[['Common Name', 'Foe']]
strain_label.drop_duplicates(inplace = True)
IVV_label = strain_label[strain_label['Common Name'].str.contains('NIST')]

def get_train_test_samples_for_baseline(train_percent, df):
    '''
    Given a % of strains to keep in the training corpus, randomly select those strains
    train_percent = float between 0-1 to use for training data
    '''
    train_df,test_df = train_test_split(df, test_size = 1-train_percent, stratify = df['Foe'].astype('str'))
    return train_df, test_df

def sweep_testharness(df,assay,feature_cols_to_use,feature_cols_to_normalize,
                      sparse_cols_to_use,output_dir,rf_default = False,repeat=20,pred_col='Foe'):
    '''
    :param df: df to run ML
    :param assay: name of the df
    :param output_dir: test-harness output dir
    :return: test-harness leaderboard in the output_dir
    '''
    th_path = output_dir
    th = TestHarness(output_location = th_path)
    percents=[0.2,0.3,0.4,0.5,0.6,0.7,0.8]
    for percent in percents:
        for i in range(repeat):
            print('#'*200)
            print(i,str(percent))
            print('#'*200)
            train_df, test_df = get_train_test_samples_for_baseline(percent,df)
            for col in feature_cols_to_use:
                train_df[col]=train_df[col].astype(str)
                test_df[col]=test_df[col].astype(str)

            normalize=False
            if feature_cols_to_normalize:
                normalize=True

            # custom run with a custom train/test split
            th.run_custom(function_that_returns_TH_model=gradient_boosted_tree, dict_of_function_parameters={}, training_data=train_df,
                          testing_data=test_df, description=assay+'__'+str(percent),
                          target_cols=pred_col, feature_cols_to_use=feature_cols_to_use,
                          index_cols=["index_col"], normalize=normalize, feature_cols_to_normalize=feature_cols_to_normalize,
                          feature_extraction=False,sparse_cols_to_use=sparse_cols_to_use,predict_untested_data=False)
            th.run_custom(function_that_returns_TH_model=gaussian_naive_bayes_classification, dict_of_function_parameters={}, training_data=train_df,
                          testing_data=test_df, description=assay+'__'+str(percent),
                          target_cols=pred_col, feature_cols_to_use=feature_cols_to_use,
                          index_cols=["index_col"], normalize=normalize, feature_cols_to_normalize=feature_cols_to_normalize,
                          feature_extraction=False,sparse_cols_to_use=sparse_cols_to_use,predict_untested_data=False)
            if rf_default == True:
                parameters = {'n_estimators':100, 'criterion':'gini', 'min_samples_leaf':1, 'max_features':'auto'}
            else:
                parameters = {}
            th.run_custom(function_that_returns_TH_model=random_forest_classification, dict_of_function_parameters=parameters, training_data=train_df,
                          testing_data=test_df, description=assay+'__'+str(percent),
                          target_cols=pred_col, feature_cols_to_use=feature_cols_to_use, 
                          index_cols=["Common Name", "index_col"], normalize=normalize, feature_cols_to_normalize=feature_cols_to_normalize,
                          feature_extraction=False,sparse_cols_to_use=sparse_cols_to_use,predict_untested_data=False)
            th.run_custom(function_that_returns_TH_model=logistic_classifier, dict_of_function_parameters={}, training_data=train_df,
                          testing_data=test_df, description=assay+'__'+str(percent),
                          target_cols=pred_col, feature_cols_to_use=feature_cols_to_use,
                          index_cols=["Common Name", "index_col"], normalize=False, feature_cols_to_normalize=feature_cols_to_normalize,
                          feature_extraction=False,sparse_cols_to_use=sparse_cols_to_use,predict_untested_data=False)


# def plot_baseline_leaderboard(assay_name, th_path):
#     '''
#     :param assay_name: assay name to query from the leaderboard
#     :param th_path: test harness path
#     :return: boxplot of the accuracy for each model
#     '''
#     sns.set(rc={'figure.figsize':(6,4)},font_scale=1.25)
#     sns.set_style("white")
#     leaderboard_df = query_leaderboard(query={'Description':assay_name},
#                                        th_output_location=th_path, loo=False, classification=True)
#     leaderboard_df[['assay', '% of Total Number of Strains In Train']] = leaderboard_df['Description'].str.split('__', expand = True)
#     leaderboard_df['% of Total Number of Strains In Train'] = ((leaderboard_df['% of Total Number of Strains In Train']).astype(float)*100).astype(int)
#     leaderboard_df['Balanced Accuracy'] = leaderboard_df['Balanced Accuracy']*100
#     model_name_dict = {'random_forest_classification':'Random Forest', 'gradient_boosted_tree':'Gradient Boosted Tree',
#                        'gaussian_naive_bayes_classification':'Gaussian Naive Bayes', 'logistic_classifier':'Logistic Regression',
#                        'support_vector_classifier':'Support Vector Machine'}
#     leaderboard_df['Model Name'] = leaderboard_df['Model Name'].map(model_name_dict)
#     for model in leaderboard_df['Model Name'].unique():
#         subset = leaderboard_df[leaderboard_df['Model Name'] == model]
#         ax = sns.boxplot(x='% of Total Number of Strains In Train', y="Balanced Accuracy",
#                          data = subset)
#         ax.set_title(model)
#         ax.set(ylim=(0,100))
#         plt.show()
#         plt.close()

def get_max_accuracy(assay, th_path, print_std = False):
    '''
    :param assay_name: assay name to query from the leaderboard
    :param th_path: test harness path
    :return: best accuracy and the standard deviation of the model
    '''
    df_leader_board = query_leaderboard(query={'Description':assay,
                                               'Column Predicted':'Foe'},
                                        th_output_location=th_path, classification=True)
    mean_accuracy = df_leader_board.groupby(['Description','Model Name']).mean()['Balanced Accuracy']*100
    max_acc = mean_accuracy[mean_accuracy == mean_accuracy.max()]
    std_accuracy = df_leader_board.groupby(['Description','Model Name']).std()['Balanced Accuracy']*100
    max_acc_std = std_accuracy[mean_accuracy == mean_accuracy.max()]
    print(round(max_acc, 1))
    if print_std == True:
        print(round(max_acc_std))
    return round(max_acc, 1), round(max_acc_std, 1)

def get_max_accuracy_description(assay, th_path):
    df_leader_board = query_leaderboard(query={'Description':assay,
                                               'Column Predicted':'Foe'},
                                        th_output_location=th_path, classification=True)
    mean_accuracy = df_leader_board.groupby(['Description','Model Name']).mean()['Balanced Accuracy']*100
    description = mean_accuracy[mean_accuracy == mean_accuracy.max()].reset_index()['Description'][0]
    model_name = mean_accuracy[mean_accuracy == mean_accuracy.max()].reset_index()['Model Name'][0]
    return description, model_name

def get_max_accuracy_no_split(assay, th_path):
    '''
    max accuracy for test harness model results without train test split at multiple percentages
    '''
    df_leader_board = query_leaderboard(query={'Description':assay,
                                               'Column Predicted':'Foe'},
                                        th_output_location=th_path, classification=True)
    mean_accuracy = df_leader_board.groupby(['Model Name']).mean()['Balanced Accuracy']*100
    print(mean_accuracy[mean_accuracy == mean_accuracy.max()])
    std_accuracy = df_leader_board.groupby(['Model Name']).std()['Balanced Accuracy']*100
    print(std_accuracy[mean_accuracy == mean_accuracy.max()])
def get_test_data(assay_name, percent, model_name, th_path):
    '''
    A function to query leaderboard and get test data in the test harness output directory
    Return test data in a df
    '''
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



def get_predict_data(assay_name, percent, model_name, th_path):
    '''
    A function to query leaderboard and get predict data in the test harness output directory
    Return predicted data in a df
    '''
    percent_df = pd.DataFrame(columns = ['% of Total Samples In Train', 'Balanced Accuracy'])
    df_leaderboard = query_leaderboard(query={'Description':assay_name+'__'+str(percent),
                                             'Model Name':model_name,
                                             'Column Predicted':'Foe'},
                                       th_output_location=th_path, classification=True)

    run_ids = df_leaderboard['Run ID'].tolist()
    predict_dfs = []
    for run_id in run_ids:
        # for loop to read test data from 4 runs
        predict_df = pd.read_csv(list(get_result_csv_paths([run_id], th_output_location=th_path,
                                                        file_type='predicted_data').keys())[0])
        predict_dfs.append(predict_df)

    predict_dfs = pd.concat(predict_dfs)
    return predict_dfs

def get_test2_data():
    '''
    extension of get_test_data 
    get test2 data more quickly 
    '''
    test2_path = r'F:\data\leave_organisms_out_20'
    baseline_path = r'F:\data\baseline_model2'
    best_adh_data2 = get_test_data('adherence', 0.2, 'gaussian', test2_path)
    best_tox_data2 = get_test_data('toxicity', 0.8, 'random', test2_path)
    best_reporter_data2 = get_test_data('reporter_live_fixed', 0.8, 'gaussian', test2_path)
    best_amr_data2 = get_test_data('amr2_mic', 0.7, 'random', test2_path)
    nist_baseline_data = get_test_data('baseline', 0.8, 'random', baseline_path)
    return best_adh_data2, best_tox_data2, best_reporter_data2, best_amr_data2, nist_baseline_data

def get_stage3_data():
    test2_data = get_test2_data()
    # aggregate foe_proba for every assay
    agg_adh, agg_tox, agg_rep, agg_amr, agg_nist = list(map(get_agg_foe_proba, test2_data))
    # concat the assay foe_proba. this method gives equal weight to each assay in the combination
    adh_tox_rep_amr = pd.concat([agg_adh, agg_tox, agg_rep, agg_amr])
    tox_rep_amr = pd.concat([agg_tox, agg_rep, agg_amr])
    tox_rep = pd.concat([agg_tox, agg_rep])
    rep_amr = pd.concat([agg_rep, agg_amr])
    tox_amr = pd.concat([agg_tox, agg_amr])
    return adh_tox_rep_amr, tox_rep_amr, tox_rep, rep_amr, tox_amr, agg_nist

def get_stage3_data_custom_reporter(reporter_data):
    test2_data = get_test2_data()
    agg_adh, agg_tox, agg_rep, agg_amr, agg_nist = list(map(get_agg_foe_proba, test2_data))
    agg_custom_rep = get_agg_foe_proba(reporter_data)
    adh_tox_rep_amr = pd.concat([agg_adh, agg_tox, agg_custom_rep, agg_amr])
    tox_rep_amr = pd.concat([agg_tox, agg_custom_rep, agg_amr])
    tox_rep = pd.concat([agg_tox, agg_custom_rep])
    rep_amr = pd.concat([agg_custom_rep, agg_amr])
    tox_amr = pd.concat([agg_tox, agg_amr]) 
    return adh_tox_rep_amr, tox_rep_amr, tox_rep, rep_amr, tox_amr, agg_nist

def get_agg_foe_proba(assay_data):
    foe_pred_df = assay_data.groupby(['Common Name'])['Foe_prob_predictions'].mean()
    return foe_pred_df.reset_index()
    
def plot_foe_prediction_heatmap(assay_data, assay_names, baseline_col, label_df = IVV_label, 
                                join_method = 'outer', main_title = None, plot_heatmap = True, return_dfs = True):
    '''
    :param assay_data: a list of assay df output from get_test_data()
    :param assay_names: a list of assay name used for heatmap labels
    :param label_df: a df contains IVV strains and the 'Foe' label
    :return: a heatmap to compare each assay with baseline
    '''
    foe_pred_dfs = []
    for data in assay_data:
        #         data = data[~data.Foe_prob_predictions.between(0.4, 0.5)]
        foe_pred_df = data.groupby(['Common Name'])['Foe_prob_predictions'].mean()
        foe_pred_dfs.append(foe_pred_df)

    joined_foe_prob = reduce(lambda left, right: pd.merge(left, right, on='Common Name', how= join_method), foe_pred_dfs)
    joined_foe_prob.columns = assay_names

    foe_label = label_df['Foe'].astype(float)
    foe_label.index = label_df['Common Name']

    joined_foe_prob = pd.merge(joined_foe_prob, foe_label, left_index=True, right_index=True)
    joined_foe_prob.rename(columns={'Foe': 'Actual'}, inplace=True)
    joined_foe_prob = joined_foe_prob.sort_values(by=['Actual', 'Common Name'])
    if plot_heatmap == True:
        sns.set(rc={'figure.figsize': (6, 12)}, font_scale=2)
        sns.heatmap(joined_foe_prob, center=0.5)
        plt.show()
        plt.close()

    quantification_df = plot_performance_quantification(joined_foe_prob, baseline_col, main_title)
    if return_dfs == True:
        return joined_foe_prob, quantification_df


def plot_performance_quantification(joined_df, baseline_col, main_title):
    '''
    :param joined_df: foe_probability from multiple assays, output from plot_foe_prediction_heatmap()
    :param baseline_col: col name used as the baseline
    :return: accuracy quantification
    '''
    cols = joined_df.columns[:-2]

    auc_list = []
    i = 1
    sns.set_style('white')
    plt.figure(figsize = (35, 6))
    
    for col in cols:
        abs_error = abs(joined_df[baseline_col] - joined_df[col])
        new_error = [a for a in abs_error if str(a) != 'nan']  # not all the strains were tested in all the assays so could produce nan score
        area = trapz(abs_error) # get auc
        full_area = len(abs_error) 
        auc = round(1- (area/full_area), 2) # get auc ratio compared to full area if the performance for all the strains are 1
        auc_list.append(auc)
        new_acc = [1 - error for error in new_error]
        plt.subplot(1, 5, i)
        plt.scatter(range(len(new_acc)), sorted(new_acc))
        plt.title(f'{col} (AUC={auc})')
        plt.ylim(0, 1)
        plt.xlabel('strains')
        plt.ylabel('Performance')
        plt.subplots_adjust(wspace=0.2, top = 0.8)
        plt.suptitle(main_title, fontsize = 22)
        i += 1
        
    return pd.DataFrame({'assay': cols, 'AUC': auc_list})

def get_all_test_data(assay_name, th_path, percent = 0.6):
    '''
    A function to query leaderboard and get test data in the test harness output directory
    Return test data in a df
    '''
    df_leaderboard = query_leaderboard(query={'Description':assay_name+'__'+str(percent),
                                              'Column Predicted':'Foe'}, 
                                       th_output_location=th_path, classification=True)

    run_ids = df_leaderboard['Run ID'].tolist()
    print(assay_name, len(run_ids))
    test_dfs = []
    for run_id in run_ids:
        # for loop to read test data from 20 runs
        test_df = pd.read_csv(list(get_result_csv_paths([run_id], th_output_location=th_path,
                                                        file_type='testing_data').keys())[0])
        test_df['Run ID'] = run_id
        test_dfs.append(test_df)
        
    test_dfs = pd.concat(test_dfs)
    df_leaderboard_short = df_leaderboard[['Run ID', 'Model Name','Description']]
    
    test_dfs = pd.merge(test_dfs, df_leaderboard_short, left_on = 'Run ID', right_on = 'Run ID')
    return test_dfs

def aggregate_all_test_accuracy(dfs, assay_names, thresh):
    '''
    A function to take test-harness dfs with accuracy from various assays and the assay names
    Return aggregated accuracy for each ml method, each assay and each organism 
    '''
    new_dfs = []
    for df in dfs:
        df_cutoff = df[~df['Foe_prob_predictions'].between(0.5 - thresh, 0.5 + thresh)]
        agg_df = df_cutoff.groupby(['Common Name','Model Name'])['Foe_prob_predictions'].mean().reset_index()        
        agg_df = agg_df.pivot('Common Name','Model Name', 'Foe_prob_predictions')
        new_dfs.append(agg_df)
    columns = list(agg_df.columns)
    fixed_columns = [f'{col}_{assay}' for assay in assay_names for col in columns ]
    df_final = reduce(lambda left,right: pd.merge(left, right, on='Common Name'), new_dfs)
    df_final.columns = fixed_columns
    return df_final

def get_stage1_for_transfer():
    ivv_features = pd.read_csv(prefix+'/Shared drives/STTR_Netrias/Project Stage/Data/ivv_features.csv')
    test2_path = 'F:/data/leave_organisms_out_20'
    adh_data = get_all_test_data('adherence', test2_path)
    tox_data = get_all_test_data('toxicity', test2_path)
    amr_data = get_all_test_data('amr2_mic', test2_path)

    agg_adh = aggregate_all_test_accuracy([adh_data], ['adh'], 0)
    agg_tox = aggregate_all_test_accuracy([tox_data], ['tox'], 0)
    agg_amr = aggregate_all_test_accuracy([amr_data], ['amr'], 0)

    # join best model aggregated results
    agg_combined = reduce(lambda left,right: pd.merge(left,right,on='Common Name'), 
                          [agg_adh['gaussian_naive_bayes_classification_adh'], 
                           agg_tox['random_forest_classification_tox'], agg_amr['random_forest_classification_amr']]) 
    agg_pred_labeled = pd.merge(agg_combined, ivv_features, left_index = True, right_on='Common Name') # add the NIST features
    agg_pred_labeled[['gaussian_naive_bayes_classification_adh', 'random_forest_classification_tox', 'random_forest_classification_amr']] *= 2
    return agg_pred_labeled

def common_name_to_genus_speices():
    '''A function to get the IVV name dictionary. Common Name as the key and genus_species as the values'''
    # strain_name_mapping = pd.read_excel(r'G:\Shared drives\STTR_Netrias\References\strain_name_mapping.xlsx')
    strain_name_mapping = pd.read_excel(prefix+'/Shared drives/STTR_Netrias/References/strain_name_mapping.xlsx')
    strain_name_mapping['Genus'] = strain_name_mapping['Genus'].str.strip()
    strain_name_mapping['Species'] = strain_name_mapping['Species'].str.strip()
    ivv_strain_names = strain_name_mapping[strain_name_mapping['Common Name'].str.contains('NIST')]
    ivv_strain_names[['Species1', 'Strain']] = ivv_strain_names['Species'].str.split(' ',1, expand=True)
    ivv_strain_names = ivv_strain_names.assign(genus_species = ivv_strain_names.Genus.str.lower() + ' ' + ivv_strain_names.Species1.str.lower())
    name_mapping = pd.Series(ivv_strain_names['genus_species'].values, index = ivv_strain_names['Common Name']).to_dict()
    return name_mapping

def genus_speices_and_foe_df():
    '''A function to get the genus_species, genus and Foe df without duplicates'''
    # strain_name_mapping = pd.read_excel(r'G:\Shared drives\STTR_Netrias\References\strain_name_mapping.xlsx')
    strain_name_mapping = pd.read_excel(prefix+'/Shared drives/STTR_Netrias/References/strain_name_mapping.xlsx')

    strain_name_mapping['Genus'] = strain_name_mapping['Genus'].str.strip()
    strain_name_mapping['Species'] = strain_name_mapping['Species'].str.strip()
    ivv_strain_names = strain_name_mapping[strain_name_mapping['Common Name'].str.contains('NIST')]
    ivv_strain_names[['Species1', 'Strain']] = ivv_strain_names['Species'].str.split(' ',1, expand=True)
    ivv_strain_names = ivv_strain_names.assign(genus_species = ivv_strain_names.Genus.str.lower() + ' ' + ivv_strain_names.Species1.str.lower())
    ivv_strain_names['genus'] = strain_name_mapping['Genus'].str.lower()
    return ivv_strain_names[['genus_species','genus','Foe']].drop_duplicates().dropna()