from tensorflow.keras import layers
from tensorflow.keras import Model, Input
from sklearn.metrics import normalized_mutual_info_score as nmi, rand_score, classification_report
from sklearn.cluster import KMeans
import tensorflow as tf
import umap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':18})

def embed_features_no_model(test_col_list, dimension = 1):
    '''
    Embed a list of encoded features and reduce to 2D with UMAP for visualization
    test_col_list is a list of cdc feature label-encoded
    '''
    embedded_list = []
    # loop through the tests and embed each test
    for test_col in test_col_list:
        num = np.unique(test_col).shape[0]
        emb_layer = layers.Embedding(num, dimension)
        embedded_test = emb_layer(test_col)
        embedded_list.append(embedded_test)
    # concatenate the embedding
    concat_layer = layers.Concatenate()
    concat_embedded = concat_layer([emb for emb in embedded_list]).numpy()
    reducer = umap.UMAP(n_neighbors = 2, min_dist = 0.8, random_state = 42)
    umap_embedding = reducer.fit_transform(concat_embedded)
    return concat_embedded, umap_embedding

# plot umap embedding 
def plot_emb_features(emb_df, index_to_annotate = [], title = '', annotation = ''):
    plt.figure(figsize = (8,6))
    plt.rcParams.update({'font.size':18})
    plt.scatter(emb_df[:,0], emb_df[:,1], s = 3)
    plt.title(title)
    if len(index_to_annotate) > 0:
        for index in index_to_annotate:
            plt.text(emb_df[index, 0]+0.2, emb_df[index, 1],annotation, 
                     horizontalalignment='left', size='xx-small', color='red', alpha = 0.6)
            
            
# embedding model to predict bacteria id 
tf.config.run_functions_eagerly(True)
def emb_model(test_col_list, label_num):
    '''
    test_col_list: a list of label-encoded features
    label_num: number of classes
    '''
    input_list = []
    emb_list = []
    for test_col in test_col_list:
        num = np.unique(test_col).shape[0]
        input_layer = layers.Input(shape=(1,))
        input_list.append(input_layer)
        emb_layer = layers.Embedding(num, 1, trainable=True)(input_layer)
        emb_list.append(emb_layer)
    concat_layer = layers.Concatenate()([emb for emb in emb_list])
    flatten_layer = layers.Flatten()(concat_layer)
    output_layer = layers.Dense(label_num, activation = 'softmax')(flatten_layer)
    model = Model(inputs=[input_layer for input_layer in input_list], outputs=output_layer)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def umap_emb_model_output(model, layer_name, data_to_predict):
    '''
    Get a specific layer from a NN model and visualize it with UMAP
    '''
    layer_to_get = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    layer_output = layer_to_get.predict(data_to_predict)
    reducer = umap.UMAP(n_neighbors = 2, min_dist = 0.8, random_state = 42)
    post_training_emb = reducer.fit_transform(layer_output)
    return layer_output, post_training_emb


def evaluate_NN(model, test_features, y_test):
    '''Evaluate ML predictions'''
    y_pred = model.predict(test_features)
    y_pred_bool = np.argmax(y_pred, axis=1) # convert from probability to class prediction
    y_test_bool = np.argmax(y_test, axis = 1) # convert from one hot encode to class
    print(classification_report(y_test_bool, y_pred_bool))
    
def quantify_clustering(cdc_data, pre_train_emb, post_train_emb):
    '''
    Compare clustering by the species ID vs clusters by kmeans in pre and post-training embedding
    '''
    num_species = cdc_data.final_id_clean.unique().shape[0]
    kmeans = KMeans(n_clusters = num_species)
    kmeans_pre_train = kmeans.fit(pre_train_emb).labels_
    kmeans_post_train = kmeans.fit(post_train_emb).labels_
    nmi_list = []
    rand_list = []
    for kmeans in [kmeans_pre_train, kmeans_post_train]:
        nmi_list.append(round(nmi(cdc_data.final_id_clean, kmeans), 2))
        rand_list.append(round(rand_score(cdc_data.final_id_clean, kmeans), 2))
        
    summary = pd.DataFrame([nmi_list, rand_list], index = ['NMI', 'RAND'], columns = ['Before', 'After'])
    return summary

def plot_emb_species(umap, sbrl_df, species):
    umap_df = pd.DataFrame(umap, columns = ['UMAP 1','UMAP 2'])
    umap_df['final_id_clean'] = sbrl_df.final_id_clean
    subset = umap_df[umap_df.final_id_clean.isin(species)]
    sns.set(font_scale = 2.5)
    sns.set_style('white')
    plt.figure(figsize = (8,6))
    g = sns.scatterplot(x='UMAP 1',y='UMAP 2', data = subset, hue = 'final_id_clean', 
                    palette = 'RdYlGn', linewidth = 0, s = 6, legend = 'full')
    g.legend(fontsize = '18')