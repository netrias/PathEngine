import pandas as pd
def get_mapping_df():
    ref_path = '../data_files/strain_name_mapping.xlsx'
    mapping_df = pd.read_excel(ref_path)
    mapping_df['Name'] = mapping_df['Name'].apply(str).apply(str.lower)
    return mapping_df