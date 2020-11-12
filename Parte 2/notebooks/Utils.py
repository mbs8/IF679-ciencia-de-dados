import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def visualize_correlation_matrix(data, hurdle = 0.0):
    R = np.corrcoef(data, rowvar=0)
    R[np.where(np.abs(R)<hurdle)] = 0.0
    heatmap = plt.pcolor(R, cmap=mpl.cm.coolwarm, alpha=0.8)
    heatmap.axes.set_frame_on(False)
    heatmap.axes.set_yticks(np.arange(R.shape[0]) + 0.5, minor=False)
    heatmap.axes.set_xticks(np.arange(R.shape[1]) + 0.5, minor=False)
    heatmap.axes.set_xticklabels(data.columns, minor=False)
    plt.xticks(rotation=90)
    heatmap.axes.set_yticklabels(data.columns, minor=False)
    plt.colorbar()
    plt.show()

def set_df_vehicles_data_types_2(df):
    """Set the categories of the vehicles dataset. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded
        
    """
    data_types = {'category': ['type', 'region', 'transmission', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'drive', 'paint_color', 'state'],
              'int': ['price', 'year', 'odometer'],
              'float': ['lat', 'long']}
    
    for key in data_types.keys():
        for elem in data_types[key]:
            df[elem] = df[elem].astype(key)

def load_vehicles_dataset_and_set_types_2(path):
    """Load the vehicles dataset in the dataframe and sets the data types.

    The dataset to be loaded should not contain null values.
    
    Args:
        path(str): path where the dataset is.
    
    Returns:
        pandas.DataFrame: the dataframe containing the vehicles dataset.
    """
    
    df = pd.read_csv(path)
    set_df_vehicles_data_types_2(df)
    return df

def set_categories_as_codes(df):
    categories = ['type', 'region', 'transmission', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'drive', 'paint_color', 'state']
    
    for category in categories:
        df[category] = df[category].cat.codes
    return df

def visualize_linear_correlation(df, columns, target):
    plt_columns = 3
    plt_rows = int(len(columns) / 3) + 1
    figs, axes = plt.subplots(plt_rows,plt_columns,figsize=(20,15))
    for i, column in enumerate(columns):
        df[[column, target]].plot.scatter(x=column, y=target,ax=axes[int(i / plt_columns),i % plt_columns])
        
def calculate_r2_and_mse(df, target):
    mse = mean_squared_error(df[[target]], predict)
    r2 = r2_score(df[[target]], predict)
    
    print("R2:      {:.3f}".format(r2))
    print("MSE:     {:.3f}".format(mse))