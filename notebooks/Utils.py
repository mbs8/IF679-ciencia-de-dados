import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

def get_decimal(f, aprox=2):
    """Gets the decimal part of a float
    
    Args:
        f (float): Real number to get the decimal part from
        aprox (int)(optional): Number of decimal digits, default is 2.
        
    Returns:
        float: Decimal part of f
    """
    f = round((f - int(f)), aprox)
    return f

def set_df_vehicles_data_types(df):
    """Set the categories of the vehicles dataset. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded
        
    """
    data_types = {'category': ['type', 'region', 'transmission', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'drive', 'size', 'paint_color', 'state'],
              'int': ['id', 'price', 'year'],
              'float': ['odometer', 'lat', 'long'],
              'object': ['url', 'description', 'vin']}
    
    for key in data_types.keys():
        for elem in data_types[key]:
            df[elem] = df[elem].astype(key)
            
def get_year_from_description(df):
    """Try to set the year of the vehicle from the description of the offer. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded
        
    """
    for idx in df[df['year'].isnull()].index:
        try:
            year = int(df.iloc[idx]['description'][0:4])
            if(0 < year < 2022):
                df.at[idx, 'year'] = year
        except:
            pass