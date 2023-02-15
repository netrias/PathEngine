import pandas as pd
from src.preprocessing.load_data import *
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from tensorflow.keras import Sequential
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from numpy import trapz
from harness.utils.parsing_results import *
from src.preprocessing.load_data import get_mapping_df

def get_train_test_samples(train_percent, data):
    '''
    Splitting loaded data to train and test set
    train_percent: percentage of the data used for training
    strain_mapping_data: output of preprocessing.load_data.strain_mapping()
    return train and test dataframes
    '''
    # these two lines remove the redundant 'Common Name' in the strain mapping file
    # so the splitting would reflect the actual splitting of different organism better
    strain_mapping_data = get_mapping_df()
    new_strain_mapping = strain_mapping_data[['Common Name', 'Foe']]
    new_strain_mapping = new_strain_mapping.drop_duplicates()
    new_strain_mapping = new_strain_mapping[new_strain_mapping['Foe'].isin([0,1])]
    # print('labels')
    # print(new_strain_mapping['Foe'].astype('str').value_counts())
    train_strain,test_strain = train_test_split(new_strain_mapping,
                                                test_size=1-train_percent,
                                                stratify=new_strain_mapping['Foe'].astype('str'))
    train_df = data[data['Common Name'].isin(train_strain['Common Name'])]
    test_df = data[data['Common Name'].isin(test_strain['Common Name'])]
    return train_df, test_df

def model(shape):
    model = Sequential()
    model.add(Dense(units = 64, activation = 'relu', input_shape = (shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(units = 64, activation = 'relu'))
    model.add(Dense(units = 128, activation = 'relu')) # keeping the layers shallow to compare with vanilla_model2 (deep) to see how much the dropping units quickly affect accuracy
    model.add(Dense(units = 1, activation = 'sigmoid'))
    return model

def plot_model_result(model, fitted_model, metrics, train, test, feature_cols, figure_name, custom_input = False, validation=False):
    '''
    plot the foe_probabilities of a model
    plot the training history
    show training and testing accuracy
    '''
    if custom_input == True:
        train_inputs = [train[col] for col in feature_cols]
        test_inputs = [test[col] for col in feature_cols]
    else:
        train_inputs = train[feature_cols]
        test_inputs = test[feature_cols]
    train_result = model.predict(train_inputs)
    bin_train_result = [1 if y > 0.5 else 0 for y in train_result]
    test_result = model.predict(test_inputs)
    bin_test_result = [1 if y > 0.5 else 0 for y in test_result]

    plt.figure(figsize=(25, 4))
    plt.subplots_adjust(wspace=0.3)
    # Plot confusion matrix for test
    plt.subplot(1, 5, 1)
    cm = confusion_matrix(test['Foe'], bin_test_result, normalize='true')
    sns.heatmap(cm, center=0.5, cmap='RdBu_r', vmin = 0, vmax = 1)
    plt.title('Confusion matrix on the test set')

    # Plot train foe proba
    plt.subplot(1, 5, 2)
    plt.hist(train_result)
    plt.xlabel('Foe probability')
    plt.title('Train foe probability')

    # Plot test foe proba
    plt.subplot(1, 5, 3)
    plt.hist(test_result)
    plt.xlabel('Foe probability')
    plt.title('Test foe probability')

    # plot loss throughout the training
    for i, metric in enumerate(metrics):
        plt.subplot(1, 5, i + 4)
        plt.plot(fitted_model.history[metric], label=str(metric))
        if validation == True:
            plt.plot(fitted_model.history[f'val_{metric}'], label=f'val_{metric}')
        plt.title(str(metric))
        plt.legend()
    plt.suptitle(figure_name, size=20)
    plt.subplots_adjust(top=0.8)

    print('accuracy on train set:', round(balanced_accuracy_score(train['Foe'], bin_train_result), 2))
    print('accuracy on test set:', round(balanced_accuracy_score(test['Foe'], bin_test_result), 2))
    
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

def get_ivv_name_mapping():
    strain_name_mapping = get_mapping_df()
    strain_name_mapping['Genus'] = strain_name_mapping['Genus'].str.strip()
    strain_name_mapping['Species'] = strain_name_mapping['Species'].str.strip()
    ivv_strain_names = strain_name_mapping[strain_name_mapping['Common Name'].str.contains('NIST')]
    ivv_strain_names[['Species1', 'Strain']] = ivv_strain_names['Species'].str.split(' ',1, expand=True)
    ivv_strain_names = ivv_strain_names.assign(genus_species = ivv_strain_names.Genus.str.lower() + ' ' + ivv_strain_names.Species1.str.lower())
    name_mapping_dict = pd.Series(ivv_strain_names['Common Name'].values, index = ivv_strain_names['genus_species']).to_dict()
    return name_mapping_dict

def get_cdc_baseline_scores():
    path = prefix+'/Shared drives/STTR_Netrias/Phase II/datadata/cdc_Feb24'
    cdc_baseline_data = get_test_data('CDC1', 0.8, 'random', path)
    baseline_score = cdc_baseline_data.groupby(['Common Name'])['Foe_prob_predictions'].mean().reset_index()
    ivv_name_mapping = get_ivv_name_mapping()
    baseline_score = baseline_score.rename(columns = {'Common Name':'genus_species'})
    baseline_score = baseline_score.assign(Name = baseline_score['genus_species'].map(ivv_name_mapping))
    baseline_score['Common Name'] = baseline_score['Name']
    baseline_score.drop(columns = ['Name', 'genus_species'], inplace = True)
    baseline_score.index = baseline_score['Common Name']
    baseline_score.drop(columns = 'Common Name', inplace = True)
    return baseline_score.dropna()

def get_nist_baseline_scores():
    path = prefix+'/Shared drives/STTR_Netrias/Phase II/data/baseline_model2'
    nist_baseline_data = get_test_data('baseline', 0.8, 'random', path)
    nist_path_score = nist_baseline_data.groupby(['Common Name'])['Foe_prob_predictions'].mean()
    return nist_path_score

def bin_cross(y_true, y_pred):
    '''To return simple binary crossentropy'''
    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return loss

def entropy_loss(p):
    if p >= 1:
        return 0
    if p <= 0:
        return 0
    else:
        logp = tf.math.log(p)
        return -tf.math.multiply(p, logp)

def get_baseline_alignment_score(joined_df):
    abs_acc = 1-abs(joined_df['NIST baseline'] - joined_df['Neural Net'])

        
def get_NN_data(model, test, feature_cols, custom_input = False):
    '''Similar to get_test_data but for NN'''
    if custom_input == True:
        test_inputs = [test[col] for col in feature_cols]
    else:
        test_inputs = test[feature_cols]   
    test_results = model.predict(test_inputs)
    test_data = test[['Common Name']]
    test_data['Foe_prob_predictions'] = test_results
    return test_data

def align_w_baseline(models, model_names, test_dfs, test_feature_dfs, baseline = 'NIST'):
    '''A function to plot a heatmap and calculate AUC to determine stage 2 performance'''
    if baseline == 'NIST':
        baseline_scores = get_nist_baseline_scores()
    elif baseline == 'CDC':
        baseline_scores = get_cdc_baseline_scores()
    else:
        print(f'{baseline} is not a recognized baseline type')
    height = test_dfs[0]['Common Name'].unique().shape[0]/3.3  
    plt.figure(figsize=(17, height))
    plt.rcParams.update({'font.size': 16})
    joined_dfs = []
    n = len(models)
    for i, model in enumerate(models):
        predictions = model.predict([test_feature_dfs[i]])
        test_df_copy = test_dfs[i].copy()
        test_df_copy['Foe_prob_predictions'] = predictions
        agg_Foe_prob = test_df_copy.groupby(['Common Name'])['Foe_prob_predictions'].mean()
        joined = pd.merge(agg_Foe_prob, baseline_scores, left_index=True, right_index=True)
        joined = joined.sort_values(['Foe_prob_predictions_y', 'Common Name'], ascending=False)

        joined.columns = ['Neural Net', 'NIST baseline']
        joined_dfs.append(joined)
        plt.subplot(1, 4, i+1)
        sns.heatmap(joined, center=0.5, yticklabels = True)
        plt.title(f'{model_names[i]} stage2')
        plt.yticks(rotation=0)
        plt.subplots_adjust(wspace=2)

    plt.figure(figsize=(20, 3))   
    for i, df in enumerate(joined_dfs):
        auc, performance_df = get_performance_auc(df)
        plt.subplot(1, 4, i+1)
        plt.scatter(range(len(df)), performance_df['Performance'].sort_values())
        plt.ylim(0, 1)
        plt.xlabel('strains')
        plt.ylabel('Performance')
        plt.title(f'AUC={auc}')
        plt.subplots_adjust(wspace=0.5)
        
def get_performance_auc(joined_df):
    '''Calculate AUC by comparing NN and NIST baseline pathogenic scores'''
    abs_error = abs(joined_df['Neural Net'] - joined_df['NIST baseline'])
    joined_df['Performance'] = 1 - abs_error
    area = trapz(abs_error)
    auc = round(1-area/len(joined_df), 2)
    return auc, joined_df

def bce_n_entropy_loss(y_true, y_pred):
    '''Binary crossentropy + entropy loss'''
    # slicing and reshaping y for proper shape to do calculation
    y_len = tf.shape(y_true)[0]
    y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
    y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # index for each strain
    # reshape so unique_with_counts will work
    y_true0 = tf.reshape(y_true0, [y_len])
    y_true1 = tf.reshape(y_true1, [y_len])
    y_pred = tf.reshape(y_pred, [y_len])
    # get binary crossentropy
    bin_cross_loss = bin_cross(y_true0, y_pred)

    # get index for the samples in the same strain
    y, idx, count = tf.unique_with_counts(y_true1)
    # calculate average foe_probability for each condition
    p_means = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=tf.size(count)) 
    e_loss = tf.map_fn(entropy_loss, p_means) # using mapping on entropy_loss1 so each strain gets its own entropy loss 
    e_loss_sum = tf.reduce_sum(e_loss) # sum the entropy loss from all strains
    
    return bin_cross_loss + e_loss_sum

def adjusted_bce(shift):
    def loss(y_true, y_pred):
        bce = tf.losses.binary_crossentropy(y_true, y_pred)
        return bce + shift
    return loss

def adjusted_bce_n_entropy_loss(shift, weight):
    def loss(y_true, y_pred):
    # slicing and reshaping y for proper shape to do calculation
        y_len = tf.shape(y_true)[0]
        y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
        y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # index for each strain
        # reshape so unique_with_counts will work
        y_true0 = tf.reshape(y_true0, [y_len])
        y_true1 = tf.reshape(y_true1, [y_len])
        y_pred = tf.reshape(y_pred, [y_len])
        # get binary crossentropy
        bin_cross_loss = bin_cross(y_true0, y_pred)
        # get index for the samples in the same strain
        y, idx, count = tf.unique_with_counts(y_true1)
        # calculate average foe_probability for each condition
        p_means = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=tf.size(count)) + shift
        e_loss = tf.map_fn(entropy_loss, p_means) # using mapping on entropy_loss1 so each strain gets its own entropy loss 
        e_loss_sum = tf.reduce_sum(e_loss) # sum the entropy loss from all strains
        return bin_cross_loss + weight*e_loss_sum
    return loss

def adjusted_bce_n_baseline_loss(shift, weight):
    def loss(y_true, y_pred):
    # slicing and reshaping y for proper shape to do calculation
        y_len = tf.shape(y_true)[0]
        y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
        y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # baseline foe proba
        # reshape so unique_with_counts will work
        y_true0 = tf.reshape(y_true0, [y_len])
        y_true1 = tf.reshape(y_true1, [y_len])
        y_pred = tf.reshape(y_pred, [y_len])
        # get binary crossentropy
        bin_cross_loss = bin_cross(y_true0, y_pred)
        # get index for the samples in the same strain
        y, idx, count = tf.unique_with_counts(y_true1)
        
        # calculate average foe_probability for each condition
        bin_p_means = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=tf.size(count)) + shift
        actual_score = tf.math.unsorted_segment_mean(data=y_true1, segment_ids=idx, num_segments=tf.size(count))
        actual_score = tf.cast(actual_score, dtype = tf.float32)
        base_loss = bin_p_means - actual_score
        base_loss = tf.map_fn(lambda x:  tf.keras.backend.abs(x), base_loss)
        base_loss_sum = tf.reduce_sum(base_loss)
        return base_loss_sum + weight*base_loss_sum 
    return loss

def baseline_loss(y_true, y_pred):
    # slicing and reshaping y for proper shape to do calculation
    y_len = tf.shape(y_true)[0]
    y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
    y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # index for each strain
    y_true2 = tf.slice(y_true, [0,2], [y_len,1]) # baseline pathagenic score
    
    # reshape so unique_with_counts will work
    y_true0 = tf.reshape(y_true0, [y_len])
    y_true1 = tf.reshape(y_true1, [y_len])
    y_true2 = tf.reshape(y_true2, [y_len])
    y_pred = tf.reshape(y_pred, [y_len])

    # get index for the samples in the same strain
    y, idx, count = tf.unique_with_counts(y_true1)
    
    # calculate pathogenic score for each strain
    bin_p_means = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=tf.size(count))
    actual_score = tf.math.unsorted_segment_mean(data=y_true2, segment_ids=idx, num_segments=tf.size(count))
    actual_score = tf.cast(actual_score, dtype = tf.float32)
    base_loss = bin_p_means - actual_score
    base_loss = tf.map_fn(lambda x:  tf.keras.backend.abs(x), base_loss)
    base_loss_sum = tf.reduce_sum(base_loss)
    return base_loss_sum

def baseline_loss_2D(y_true, y_pred):
    # baseline loss with 2D y: FoF label and NIST-derived pathogenic score
    # slicing and reshaping y for proper shape to do calculation
    y_len = tf.shape(y_true)[0]
    y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
    y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # index for each strain
    
    # reshape so unique_with_counts will work
    y_true0 = tf.reshape(y_true0, [y_len])
    y_true1 = tf.reshape(y_true1, [y_len])
    y_pred = tf.reshape(y_pred, [y_len])

    # get index for the samples in the same strain
    y, idx, count = tf.unique_with_counts(y_true1)
    
    # calculate pathogenic score for each strain
    bin_p_means = tf.math.unsorted_segment_mean(data=y_pred, segment_ids=idx, num_segments=tf.size(count))
    actual_score = tf.math.unsorted_segment_mean(data=y_true1, segment_ids=idx, num_segments=tf.size(count))
    actual_score = tf.cast(actual_score, dtype = tf.float32)
    base_loss = bin_p_means - actual_score
    base_loss = tf.map_fn(lambda x: tf.keras.backend.abs(x), base_loss)
    base_loss_sum = tf.reduce_sum(base_loss)
    return base_loss_sum

def bce_n_baseline_loss(y_true, y_pred):
    '''Binary crossentropy + baseline loss'''
    y_len = tf.shape(y_true)[0]
    y_true0 = tf.slice(y_true, [0,0], [y_len,1]) # actual label
    y_true1 = tf.slice(y_true, [0,1], [y_len,1]) # index for each strain
    # reshape so unique_with_counts will work
    y_true0 = tf.reshape(y_true0, [y_len])
    y_true1 = tf.reshape(y_true1, [y_len])
    y_pred = tf.reshape(y_pred, [y_len])
    # get binary crossentropy
    bin_cross_loss = bin_cross(y_true0, y_pred)

    baseline_loss_sum = baseline_loss(y_true, y_pred)
    return bin_cross_loss + baseline_loss_sum

def plot_accuracy(accuracy_df, title):
    '''
    input accuracy df with percent split and accuracy scores
    output boxplot split by percent
    '''
    sns.set(rc={'figure.figsize':(6,4)},font_scale=1.5)
    sns.set_style("white")
    accuracy_copy = accuracy_df.copy()
    accuracy_copy['percent'] = (accuracy_copy['percent']*100).astype(int)
    ax = sns.boxplot(x='percent', y="accuracy", 
                     data = accuracy_copy)
    ax.set_title(title)
    ax.set(ylim=(0,110))
    plt.axhline(50, color = 'r')
    plt.ylabel('Balanced Accuracy')
    plt.xlabel('% of Total Number of Strains In Train')
    
def get_accuracy(model, test_df, features):
    prediction = model.predict(test_df[features])
    y_pred = [1 if y > 0.5 else 0 for y in prediction]
    accuracy = round(100*balanced_accuracy_score(test_df['Foe'], y_pred), 2)
    return accuracy

def add_baseline_score(assay_df):
    assay_copy = assay_df.copy()
    '''Add baseline pathogenic score to the assay df for baseline loss'''
    baseline_scores = get_nist_baseline_scores()
    assay_copy = pd.merge(assay_copy, baseline_scores, left_on = 'Common Name', right_on = 'Common Name')
    return assay_copy

def run_custom_model(loss_function, train, test, feature_cols, y_cols, epochs = 50, batch_size = 32, 
                     output_dir = None, model_name = None):
    model1 = model(len(feature_cols))
    model1.compile(loss = loss_function, optimizer='Adam', metrics=[loss_function], run_eagerly=True)
    fitted_model1 = model1.fit(train[feature_cols], train[y_cols], epochs = epochs, batch_size = batch_size, verbose = 0)
    result = model1.predict(test[feature_cols])
    if output_dir != None:
        assert model_name != None, 'Need the model name to save the model'
        model1.save(f'{output_dir}/{model_name}.h5')
    return result

def run_nn_return_test_df(assay_df, feature_cols, loss_function, percent, repeat = 16, output_dir = None, output_name = None):
    total_test_df = []
    for i in range(repeat):
        train_df, test_df = get_train_test_samples(percent, assay_df)
        result1 = run_custom_model(loss_function, train_df, test_df, feature_cols, 
                                   'Foe', epochs = 50, batch_size = 32)
        test_df['foe_proba'] = result1
        total_test_df.append(test_df)
    total_test_df = pd.concat(total_test_df).reset_index(drop = True)
    total_test_df['Foe_prob_predictions'] = total_test_df['foe_proba']
    total_test_df.drop(columns = ['foe_proba'], inplace = True)
    total_test_df['Foe_pred'] = total_test_df['Foe_prob_predictions'].apply(lambda x:0 if x < 0.5 else 1)
    print('accuracy=', round(balanced_accuracy_score(total_test_df['Foe'], total_test_df['Foe_pred']), 2))
    if output_dir:
        if output_name == None:
            print('output_name cannot be none')
        else:
            total_test_df.to_csv(output_dir + output_name, index = 0)
            print('the model has been saved')
    return total_test_df

def run_nn_return_test_df2(train_df, test_df, feature_cols, y_cols, loss_function, repeat = 16, output_dir = None, output_name = None):
    '''
    This method is different from run_nn_return_test_df in that it does not perform train test split
    on the assay df
    '''
    total_test_df = []
    for i in range(repeat):
        result1 = run_custom_model(loss_function, train_df, test_df, feature_cols, 
                                   y_cols, epochs = 50, batch_size = 32)
        test_df['foe_proba'] = result1
        total_test_df.append(test_df)
    total_test_df = pd.concat(total_test_df).reset_index(drop = True)
    total_test_df['Foe_prob_predictions'] = total_test_df['foe_proba']
    total_test_df.drop(columns = ['foe_proba'], inplace = True)
    total_test_df['Foe_pred'] = total_test_df['Foe_prob_predictions'].apply(lambda x:0 if x < 0.5 else 1)
    print(round(balanced_accuracy_score(total_test_df['Foe'], total_test_df['Foe_pred']), 2))
    if output_dir:
        if output_name == None:
            print('output_name cannot be none')
        else:
            total_test_df.to_csv(output_dir + output_name, index = 0)
            print('the model has been saved')
    return total_test_df


def sweep_NN(assay_df, feature_cols, loss_function, y_cols, percents = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], repeat = 3):
    acc_dfs = []
    for p in percents:
        acc = []
        for i in range(repeat):
            train_df, test_df = get_train_test_samples(p, assay_df)    
            result = run_custom_model(loss_function, train_df, test_df, feature_cols, 
                                      y_cols, epochs = 50, batch_size = 32)    
            bin_result = [1 if y > 0.5 else 0 for y in result]
            acc.append(100*balanced_accuracy_score(test_df['Foe'], bin_result))
        acc_df = pd.DataFrame({'accuracy':acc})
        acc_df['percent'] = p
        acc_dfs.append(acc_df)
    acc_dfs = pd.concat(acc_dfs)
    return acc_dfs

def plot_sweep_nn_accuracy(accuracy_df, model_name, print_acc = False):
    sns.set(rc={'figure.figsize': (6, 4)}, font_scale=2)
    sns.set_style("white")
    accuracy_df_copy = accuracy_df.copy()
    accuracy_df_copy['percent'] = (accuracy_df_copy['percent']*100).astype(int)
    ax = sns.boxplot(x='percent', y="accuracy", 
                     data = accuracy_df_copy)
    ax.set_title(model_name)
    if print_acc == True:
        means = accuracy_df_copy.groupby(['percent'])['accuracy'].mean().values
        std = accuracy_df_copy.groupby(['percent'])['accuracy'].std().values
        for xtick in ax.get_xticks():
            if (std[xtick] > 0.3*means[xtick]) & (means[xtick] + 30 < 80):
                space = 30
            elif (means[xtick] + 15 > 98):
                space = 5
            else:
                space = 20
            ax.text(xtick, means[xtick] + space, int(means[xtick]), 
                    horizontalalignment='center',size='medium',color='r')
    ax.set(ylim=(0,100))
    plt.axhline(50, color = 'r')
    plt.ylabel('Balanced Accuracy')
    plt.xlabel('% Strains In Train')
    
    plt.show()
    plt.close()
    
def run_and_save_model(train_df, test_df, feature_cols, loss_function, y_cols, output_dir, model_name):
    result1 = run_custom_model(loss_function, train_df, test_df, feature_cols, 
                               y_cols = y_cols, epochs = 50, batch_size = 32,
                               output_dir = output_dir, 
                               model_name = model_name)
    test_df['Foe_prob_predictions'] = result1
    test_df['Foe_pred'] = test_df['Foe_prob_predictions'].apply(lambda x:0 if x < 0.5 else 1)
    print(round(balanced_accuracy_score(test_df['Foe'], test_df['Foe_pred']), 2))

    
    
    
    
def run_nn_return_test_df2(train_df, test_df, feature_cols, y_cols, loss_function, repeat = 16, output_dir = None, output_name = None):
    '''
    This method is different from run_nn_return_test_df in that it does not perform train test split
    on the assay df
    '''
    total_test_df = []
    for i in range(repeat):
        result1 = run_custom_model(loss_function, train_df, test_df, feature_cols, 
                                   y_cols, epochs = 50, batch_size = 32)
        test_df['foe_proba'] = result1
        total_test_df.append(test_df)
    total_test_df = pd.concat(total_test_df).reset_index(drop = True)
    total_test_df['Foe_prob_predictions'] = total_test_df['foe_proba']
    total_test_df.drop(columns = ['foe_proba'], inplace = True)
    total_test_df['Foe_pred'] = total_test_df['Foe_prob_predictions'].apply(lambda x:0 if x < 0.5 else 1)
    print(round(balanced_accuracy_score(total_test_df['Foe'], total_test_df['Foe_pred']), 2))
    if output_dir:
        if output_name == None:
            print('output_name cannot be none')
        else:
            total_test_df.to_csv(output_dir + output_name, index = 0)
            print('the model has been saved')
    return total_test_df
    

def sweep_NN_w_entropy(control_df, assay_df, feature_cols, loss_function, y_cols, percents = [0.2,0.3,0.4,0.5,0.6,0.7,0.8], 
                       repeat = 3, output_dir = None, output_name = None):
    acc_dfs = []
    total_test_df = []
    for p in percents:
        percent_test_df = []
        acc = []
        for i in range(repeat):
            train_df, test_df = get_train_test_samples(p, assay_df)    
            result = run_custom_model(loss_function, control_df, test_df, feature_cols, 
                                      y_cols, epochs = 50, batch_size = 32)    
    
            bin_result = [1 if y > 0.5 else 0 for y in result]
            test_df['Foe_prob_predictions'] = result 
            test_df['Foe_pred'] = bin_result
            percent_test_df.append(test_df)
            acc.append(100*balanced_accuracy_score(test_df['Foe'], bin_result))

        percent_test_df = pd.concat(percent_test_df)
        percent_test_df['percent'] = p
        total_test_df.append(percent_test_df)
        acc_df = pd.DataFrame({'accuracy':acc})
        acc_df['percent'] = p
        acc_dfs.append(acc_df)
    acc_dfs = pd.concat(acc_dfs)
    total_test_df = pd.concat(total_test_df).reset_index(drop = True)
    if output_dir:
        assert output_name, 'output_name cannot be none'
        total_test_df.to_csv(os.path.join(output_dir, output_name), index = 0)
        print('the test data has been saved')
    return total_test_df, acc_dfs

def run_nn_return_test_w_split(train_df, test_df, feature_cols, y_cols, loss_function, percents = [0.3,0.5,0.7], repeat = 16, output_dir = None, output_name = None):
    total_test_df = []
    for p in percents:
        for i in range(repeat):
            train, test = get_train_test_samples(p, test_df)    
            result1 = run_custom_model(loss_function, train_df, test, feature_cols, 
                                          y_cols, epochs = 50, batch_size = 32)    
            test['foe_proba'] = result1
            test['percent'] = p
            test['repeat'] = i+1
            total_test_df.append(test)
    total_test_df = pd.concat(total_test_df).reset_index(drop = True)
    total_test_df['Foe_prob_predictions'] = total_test_df['foe_proba']
    total_test_df.drop(columns = ['foe_proba'], inplace = True)
    total_test_df['Foe_pred'] = total_test_df['Foe_prob_predictions'].apply(lambda x:0 if x < 0.5 else 1)
    print(round(balanced_accuracy_score(total_test_df['Foe'], total_test_df['Foe_pred']), 2))
    if output_dir:
        if output_name == None:
            print('output_name cannot be none')
        else:
            total_test_df.to_csv(output_dir + output_name, index = 0)
            print('the model has been saved')
    return total_test_df

def search_NN_entropy_loss(control_df, test_df, feature_cols):
    '''A simple grid search for 2 shift values and 2 weight values to tune entropy loss'''
    acc_df_list = []
    for shift in [0,0.1]:
        for weight in [1,1.5]:
            acc_df = sweep_NN_w_entropy(control_df, test_df, feature_cols, 
                                        adjusted_bce_n_entropy_loss(shift, weight), y_cols = ['Foe', 'org_index'], percents = [0.3,0.5,0.7], repeat = 3)
            acc_df_list.append(acc_df)
            grouped_mean = acc_df.groupby(['percent'])['accuracy'].mean()
            print(grouped_mean[grouped_mean == max(grouped_mean)])
    return acc_df_list