import pandas as pd
import numpy as np
import re
from Constants import *
from geopy.geocoders import Nominatim
#from matplotlib import pyplot as plt
#import seaborn as sns

def floatStr(decimal, decimalPlaces=2):
    """Format the decimal number into a string to be printed
    
    Args:
        decimal (float): Real number to be formatted
        decimalPlaces (int)(optional): Amount of decimal places
        
    Returns:
        String: The number formatted as string
    """
    return str(round(decimal, decimalPlaces))

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
    data_types = {'category': ['type', 'region', 'transmission', 'manufacturer', 'model', 'condition', 'cylinders', 'fuel', 'title_status', 'drive', 'paint_color', 'state'],
              'int': ['price', 'year', 'odometer'],
              'float': ['lat', 'long'],
              'object': ['url', 'description']}
    
    for key in data_types.keys():
        for elem in data_types[key]:
            df[elem] = df[elem].astype(key)

def load_vehicles_dataset_and_set_types(path):
    """Load the vehicles dataset in the dataframe and sets the data types.

    The dataset to be loaded should not contain null values.
    
    Args:
        path(str): path where the dataset is.
    
    Returns:
        pandas.DataFrame: the dataframe containing the vehicles dataset.
    """
    
    df = pd.read_csv(path)
    set_df_vehicles_data_types(df)
    return df
            
def get_brand_from_description(df):
    """Try to set the manufacturer of the vehicle based on description
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded   
    """
    brands = df['manufacturer'].unique()
    
    ##Remove Rover, Ram, Mini from the analysis - these words can make it harder
    brands = np.delete(brands, [27,19,6]) 
    
    for idx in df[df['manufacturer'] == 'undefined'].index:
        finalList = []
        for brand in brands: 
            description = df.iloc[idx]['description'][0:100]
            matchObj = re.search(brand,description,re.IGNORECASE)

            if matchObj:
                finalList.append(brand)
        if(len(finalList) == 1):
             df.at[idx, 'manufacturer'] = finalList[0]
            
def get_year_from_description(df):
    """Try to set the year of the vehicle from the description of the offer. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): Dataframe where the vehicles dataset is loaded
        
    """
    for idx in df[df['year'].isnull()].index:
        try:
            year = int(df.loc[idx]['description'][0:4])
            if(0 < year < 2022):
                df.at[idx, 'year'] = year
        except:
            pass

def get_odometer_from_description(df):
    """Try to set the odometer column of the vehicle from the information in the description. The operation is inplace.
    
    Args:
        df (pandas.DataFrame): DataFrame where the vehicles dataset is loaded.
    """
    miles_df = df[df['odometer'].isnull() & df['description'].str.contains('[Mm][Ii][Ll][Ee][Ss]*|[Mm][Ii][Ll][Ee][Aa][Gg][Ee]')]
    erros = 0
    
    for idx, row in miles_df.iterrows():
        description = df.loc[idx]['description']
        mile = re.findall(r'[Mm][Ii][Ll][Ee][Aa][Gg][Ee]:\s*\d+[Kk,.]*\d*|\d+[Kk,.]*\d*\s*[Mm][Ii][Ll][Ee][Ss]', description)

        if(mile != None and len(mile) >= 1):
            aux = 0
            mile = [re.findall(r'\d+[Kk,.]*\d*', item) for item in mile]
            mile = [format_mile_info(item[0]) for item in mile]
            try:
                for item in mile:
                    i_item = int(item)
                    if 0 < i_item < 1000000 and i_item > aux:
                        aux = i_item
                df.at[idx, 'odometer'] = aux
            except:
                erros += 1
                
    if(erros > 0):
        print("Não foi possível encontrar as milhas do carro na descrição em {} amostras.".format(erros))
        
def format_mile_info(n_str):
    """Format the mileage information extracted from the description.
    
    Args:
        n_str (str): extracted miles information from description in one line of the dataframe.
    
    Returns:
        str: formatted string to be converted in float
    """
    if(n_str[-1] == 'k' or n_str[-1] == 'K'):
        n_str = n_str[:-1] + '000'
    n_str = re.sub('[,.]', '', n_str)
    return n_str

def get_states_latitude_longitude(country="USA", states=STATES_DICT.keys()):
    """Given a list of states of a specific country, returns a dictionary containing the coordinates of each state (latitude and longitude).
    
    Args:
        country (str)(optional): country where the states belongs.
        
        states (list(str))(optional): states which to get the coordinates information.
        
    Return:
        dict: where the keys are the initials of the state and the value is a list of 2 elements:
            element[0]: latitude
            element[1]: longitude
    """
    geolocator = Nominatim(user_agent='Chrome')
    coordinates = {}
    for state in states:
        location = geolocator.geocode("{},{}".format(state, country))
        coordinates[STATES_DICT[state]] = [round(location.latitude, 4), round(location.longitude, 3)]

    return coordinates

def put_coordinates_in_dataframe(df):
    """Given the vehicle dataframe, replace the null coordinates with the information returned from get_states_latitude_longitude function.
    
    Args:
        df (pandas.DataFrame): DataFrame to replace the null coordinates.
    """
    coordinates = get_states_latitude_longitude()
    null_coordinates_df = df[df['lat'].isnull() | df['long'].isnull()]

    for idx, line in null_coordinates_df.iterrows():
        df.at[idx, 'lat'] = coordinates[line['state']][0]
        df.at[idx, 'long'] = coordinates[line['state']][1]
        
def evaluate_df_z_score_for_column(df, column):
    """Evaluate a DataFrame that all the lines in the 'column' have a z-score less than 3.5.
    
    Args:
        df (pandas.DataFrame): DataFrame to evaluate.
        
        column (str): column name where the z-score will be executed.
        
    Returns:
        pandas.DataFrame: A dataframe where every line has a z-score less than 3.5 in determined column.
    """
    mad = df[df[column] > 0][column].mad()
    z_score_df = df[abs(df[column] - df[column].median())/mad < 3.5]
    return z_score_df

def get_samples_in_usa(df):
    """Return a dataframe coitaining only samples that are in the USA territory.
        
    Args:
        df (pandas.DataFrame): dataframe where the operation will be done.
    
    Returns:
        pandas.DataFrame: slice of the dataframe containing only samples in the USA territory.
    """
    return df[(df['lat'] >= min(BOTTOM_LEFT_PT[0], BOTTOM_RIGHT_PT[0])) 
              & (df['lat'] <= max(TOP_LEFT_PT[0], TOP_RIGHT_PT[0])) 
              & (df['long'] >= min(TOP_LEFT_PT[1], BOTTOM_LEFT_PT[1]))
              & (df['long'] <= max(TOP_RIGHT_PT[1], BOTTOM_RIGHT_PT[1]))]