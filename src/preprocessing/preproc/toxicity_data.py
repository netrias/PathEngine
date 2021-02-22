import pandas as pd
from ..reference import fof_strain_mapping

def load(dapi_path, hoechst_path, add_time=False):
    '''
    New method to load toxicity data to get raw dapi and hoechst
    Sometimes TAMU uses PI for DAPI so use the PI_path for dapi_path
    '''
    strain_mapping_data = fof_strain_mapping.get_mapping_df()
    
    dapi_file = pd.read_csv(dapi_path)
    hoechst_file = pd.read_csv(hoechst_path)
    # The following 2 lines are only there for the specific df format used here. If anything changes, these need to change
    dapi_file = dapi_file.iloc[1:, 2:]
    hoechst_file = hoechst_file.iloc[1:, 2:]
    col_names = dapi_file.columns.tolist()
    # Time variable should be present in the dataframe but in this case it was missing so I manually added it back
    if add_time == True:
        dapi_file['Time'] = range(19)
        hoechst_file['Time'] = range(19)
    dapi_df = pd.melt(dapi_file, id_vars='Time', value_vars=col_names, value_name='DAPI')
    hoechst_df = pd.melt(hoechst_file, id_vars='Time', value_vars=col_names, value_name='Hoechst')

    joined = pd.merge(dapi_df, hoechst_df, left_on=['Time', 'variable'], right_on=['Time', 'variable'])
    joined['variable'] = joined['variable'].str.replace('\.1', '')
    joined['variable'] = joined['variable'].str.replace('\.2', '')
    joined['variable'] = joined['variable'].str.replace('\.3', '')
    joined.variable = joined.variable.str.lower()
    joined_labeled = pd.merge(joined, strain_mapping_data, left_on='variable', right_on='Name')
    joined_labeled = joined_labeled[joined_labeled.Foe != 'no label']

    for col in ['Time', 'DAPI', 'Hoechst']:
        joined_labeled[col] = pd.to_numeric(joined_labeled[col])
        joined_labeled = joined_labeled.sort_values(['Foe', 'variable'])
    joined_labeled.Foe = joined_labeled.Foe.astype(int)
    return joined_labeled

