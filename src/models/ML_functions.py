''''
This file will be used to run ML on loaded data with customized models and parameters
'''

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
import pandas as pd
from harness.test_harness_class import TestHarness
from harness.utils.parsing_results import *
from ..preprocessing.reference import fof_strain_mapping
import seaborn as sns

def get_train_test_samples1(train_percent, df, col_to_stratify = None):
    '''
    This is train_test_split for test 1
    Given a % of data points to keep in the each organism in the training corpus, randomly select those strains
    train_percent = float between 0-1 to use for training data
    col_to_stratify: columns included to be split the same ratio as the train test split. e.g. 'Time_Window' in toxicity data
    '''
    # In this case, no matter what percentage training is, there will be equal ratio of friends to foe in both training and testing
    strain_mapping_data = fof_strain_mapping.get_mapping_df()
    train_dfs = []
    test_dfs = []
    for strain in strain_mapping_data['Common Name'].unique():
        if len(df[df['Common Name'] == strain]) > 0:
            if col_to_stratify != None:
                train_temp, test_temp = train_test_split(df[df['Common Name'] == strain], test_size = 1-train_percent,
                                                         stratify = df[df['Common Name'] == strain][col_to_stratify])
            else:
                train_temp, test_temp = train_test_split(df[df['Common Name'] == strain], test_size = 1-train_percent)
            train_dfs.append(train_temp)
            test_dfs.append(test_temp)
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)
    return train_df, test_df
def get_train_test_samples2(train_percent, data):
    '''
    This is train_test_split for test 2
    Splitting loaded data to train and test set
    train_percent: percentage of the data used for training
    return train and test dataframes
    '''
    # these two lines remove the redundant 'Common Name' in the strain mapping file
    # so the splitting would reflect the actual splitting of different organism better
    strain_mapping_data = fof_strain_mapping.get_mapping_df()
    new_strain_mapping = strain_mapping_data[['Common Name', 'Foe']]
    new_strain_mapping = new_strain_mapping.drop_duplicates()
    train_strain,test_strain = train_test_split(new_strain_mapping,
                                                test_size=1-train_percent,
                                                stratify=new_strain_mapping['Foe'].astype('str'))
    train_df = data[data['Common Name'].isin(train_strain['Common Name'])]
    test_df = data[data['Common Name'].isin(test_strain['Common Name'])]
    return train_df, test_df

def run_ml_test1(processed_data, assay_name, output_dir,
           feature_cols_to_use, models_to_test, rf_default = False,
           repeat=4, feature_cols_to_normalize=None,
           sparse_cols_to_use=None, pred_col='Foe', percents=[0.7], stratify_cols = ['Foe'],
           predict_untested_data = False, save_trained_model = False):
    '''
    Run machine learning on preprocessed data from preprocessing module
    return: Test-harness leaderboard
    processed_data: data preprocessed in the preprocessing module
    assay_name: name of the assay, used in the leaderboard
    feature_cols_to_use: columns to be used as features for ml training in the dataframe
    models_to_test: a list of models to test
    output_dir: where test-harness leaderboard will be
    repeat: the number of times to repeat the models with random train_test_split
    feature_cols_to_normalize: feature columns to normalize
    sparse_cols_to_use: feature columns treated as sparse columns
    pred_col: label
    percent: a list of percentage of the data used in the training
    '''
    th_path = output_dir
    th = TestHarness(output_location = th_path)
    for percent in percents:
        for i in range(repeat):
            train_df, test_df = get_train_test_samples1(percent, processed_data, col_to_stratify = stratify_cols)
            for col in feature_cols_to_use:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)
            normalize = False
            if feature_cols_to_normalize:
                normalize = True

            for model in models_to_test:
                if rf_default == True and model.__name__ == 'random_forest_classification':
                    parameters = {'n_estimators':100, 'criterion':'gini', 'min_samples_leaf':1, 'max_features':'auto'}
                else:
                    parameters = {}
                th.run_custom(function_that_returns_TH_model = model,
                              dict_of_function_parameters=parameters, training_data = train_df,
                              testing_data = test_df, description = assay_name + '__'+str(percent),
                              target_cols=pred_col, feature_cols_to_use = feature_cols_to_use,
                              index_cols = ['index_col', 'Common Name'], normalize = normalize,
                              feature_cols_to_normalize = feature_cols_to_normalize, feature_extraction = False,
                              sparse_cols_to_use=sparse_cols_to_use, predict_untested_data=predict_untested_data, save_trained_model = save_trained_model)

                
def run_ml(processed_data, assay_name, output_dir,
           feature_cols_to_use, models_to_test, rf_default = False,
           repeat=4, feature_cols_to_normalize=None,
           sparse_cols_to_use=None, pred_col='Foe', percents=[0.7], 
           predict_untested_data = False, save_trained_model = False):
    '''
    Run machine learning on preprocessed data from preprocessing module
    return: Test-harness leaderboard
    processed_data: data preprocessed in the preprocessing module
    assay_name: name of the assay, used in the leaderboard
    feature_cols_to_use: columns to be used as features for ml training in the dataframe
    models_to_test: a list of models to test
    output_dir: where test-harness leaderboard will be
    repeat: the number of times to repeat the models with random train_test_split
    feature_cols_to_normalize: feature columns to normalize
    sparse_cols_to_use: feature columns treated as sparse columns
    pred_col: label
    percent: a list of percentage of the data used in the training
    '''
    th_path = output_dir
    th = TestHarness(output_location = th_path)
    for percent in percents:
        for i in range(repeat):
            train_df, test_df = get_train_test_samples2(percent, processed_data)
            for col in feature_cols_to_use:
                train_df[col] = train_df[col].astype(str)
                test_df[col] = test_df[col].astype(str)
            normalize = False
            if feature_cols_to_normalize:
                normalize = True

            for model in models_to_test:
                if rf_default == True and model.__name__ == 'random_forest_classification':
                    parameters = {'n_estimators':100, 'criterion':'gini', 'min_samples_leaf':1, 'max_features':'auto'}
                else:
                    parameters = {}
                th.run_custom(function_that_returns_TH_model = model,
                              dict_of_function_parameters=parameters, training_data = train_df,
                              testing_data = test_df, description = assay_name + '__'+str(percent),
                              target_cols=pred_col, feature_cols_to_use = feature_cols_to_use,
                              index_cols = ['index_col', 'Common Name'], normalize = normalize,
                              feature_cols_to_normalize = feature_cols_to_normalize, feature_extraction = False,
                              sparse_cols_to_use=sparse_cols_to_use, predict_untested_data=predict_untested_data, save_trained_model = save_trained_model)

def plot_leaderboard(assay, th_path, output_dir = None):
    model_dict = {'random_forest_classification': 'Random Forest',
                  'gaussan_naive_bayes_classification': 'Gaussian Naive Bayes',
                  'gaussian_naive_bayes_classification': 'Gaussian Naive Bayes',
                  'gradient_boosted_tree': 'Gradient Boosted Tree',
                  'logistic_classifier': 'Logistic Regression',
                  'support_vector_classifier': 'Support Vector Machine',
                  '__init__':'Neural Network'}
    sns.set(rc={'figure.figsize': (6, 4)}, font_scale=1.5)
    sns.set_style("white")

    i = 1
    for model in model_dict:
        df_leaderboard = query_leaderboard(query={'Description':assay,'Model Name':model},
                                           th_output_location=th_path, loo=False, classification=True)
        if len(df_leaderboard) != 0:
            df_leaderboard['percent'] = df_leaderboard['Description'].str.split(pat='__', expand=True)[1]
            df_leaderboard['% of Total Number of Strains In Train'] = df_leaderboard['percent'].astype('float')*100
            df_leaderboard['% of Total Number of Strains In Train'] = df_leaderboard['% of Total Number of Strains In Train'].astype(int)
            df_leaderboard['Balanced Accuracy'] = df_leaderboard['Balanced Accuracy'] * 100
            df_leaderboard['Model Name'] = df_leaderboard['Model Name'].map(model_dict)
            ax = sns.boxplot(x='% of Total Number of Strains In Train', y="Balanced Accuracy",
                        data=df_leaderboard.sort_values(by=['percent']))

            ax.set(ylim = (0,100))
            plt.title(model_dict[model])
            plt.axhline(50, color = 'r')
            if output_dir != None:
                plt.savefig(f'{output_dir}/{assay}_{model}.png', bbox_inches = 'tight')
                print(f'Figure {i} has been saved.')
            plt.show()
            plt.close()
            i += 1

            
def get_max_accuracy(assay, th_path):
    '''
    :param assay_name: assay name to query from the leaderboard
    :param th_path: test harness path
    :return: best accuracy and the standard deviation of the model
    '''
    df_leader_board = query_leaderboard(query={'Description':assay,
                                               'Column Predicted':'Foe'},
                                        th_output_location=th_path, classification=True)
    mean_accuracy = df_leader_board.groupby(['Description','Model Name']).mean()['Balanced Accuracy']*100
    print(mean_accuracy[mean_accuracy == mean_accuracy.max()])
    std_accuracy = df_leader_board.groupby(['Description','Model Name']).std()['Balanced Accuracy']*100
    print(std_accuracy[mean_accuracy == mean_accuracy.max()])
