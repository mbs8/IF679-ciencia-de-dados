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