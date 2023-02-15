import pandas as pd
import seaborn as sns


def plot_assay_group_by_foe(assay_data, cols_to_use):
    assay_value_cnt_dfs = []
    for col in cols_to_use:
        value_cnt = assay_data.groupby(['Foe'])[col].value_counts(normalize = True)
        value_cnt.index.names = ['Foe', 'type'] # rename index so could reset_index
        value_cnt = value_cnt.reset_index()
        value_cnt['test'] = col # add a new column for the test type
        value_cnt.rename(columns = {col:'values'}, inplace=True) # rename the original column to values
        assay_value_cnt_dfs.append(value_cnt)
    return pd.concat(assay_value_cnt_dfs).reset_index(drop = True)


def draw_heatmap(*args, **kwargs):
    sns.set(font_scale = 1.25)
    sns.set_style('white')
    data = kwargs.pop('data')
    d = data.pivot(index=args[1], columns=args[0], values=args[2])
    sns.heatmap(d, **kwargs)