import pandas as pd
from ..reference import fof_strain_mapping

def add_amr_derived_features(amr_df):
    amr_df_copy = amr_df.copy()
    amr_df_copy['DD^2'] = amr_df_copy['DD'].apply(lambda x: x ** 2)

    amr_df_copy['DD^2_std'] = 0
    for org in amr_df_copy['Common Name'].unique():
        for antibiotic in amr_df_copy['antibiotic'].unique():
            subset = amr_df_copy[(amr_df_copy['Common Name'] == org) & (amr_df_copy['antibiotic'] == antibiotic)]
            subset_std = subset['DD^2'].std()
            amr_df_copy.loc[((amr_df_copy['Common Name'] == org) & (
                        amr_df_copy['antibiotic'] == antibiotic)), 'DD^2_std'] = subset_std

    return amr_df_copy

def load(amr_file, antibiotics = []):
    '''
    amr_file: The amr data, will change this to a directory if file numbers increase
    MIC: Minimal inhibitory concentration as the model feature. This is the default setting.
    return: Dataframe ready for analysis for the adherence model
    '''
    return normalize(pd.read_excel(amr_file), antibiotics)

def normalize(amr_df, antibiotics = []):
    strain_mapping_df = fof_strain_mapping.get_mapping_df()
    amr_df['Strain ID'] = amr_df['Strain ID'].apply(str).apply(str.lower)

    amr_labeled = pd.merge(amr_df, strain_mapping_df, how='inner', right_on='Name', left_on='Strain ID')

    #antibiotics = ['Kanamycin', 'Ampicilin', 'Tetracycline', 'Chloramphenicol', 'Polymyxin B', 'Ceftazidime']
    antibiotic_cols = [ab + suf for ab in antibiotics for suf in ['_1', '_2', '_3']]
    antibiotic_proc_cols = [ab + '_Proc' for ab in antibiotics]
    # We don't analyze SIR data anymore so only DD is used
    id_cols = list(set(amr_labeled.columns) - set(antibiotic_cols))
    df_amr_melt = pd.melt(amr_labeled, id_vars=id_cols, value_vars=antibiotic_cols, var_name='antibiotic',
                             value_name='DD')
    df_amr_melt['antibiotic'] = df_amr_melt['antibiotic'].str[:-2]
    df_amr_w_extra_features = add_amr_derived_features(df_amr_melt)
    df_amr_w_extra_features['index_col'] = range(len(df_amr_w_extra_features))
    df_amr_w_extra_features['Foe'] = df_amr_w_extra_features['Foe'].astype(int)
    return df_amr_w_extra_features