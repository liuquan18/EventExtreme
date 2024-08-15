#%%
import pandas as pd
import numpy as np

# %%
import src.extreme_extract as ee
import src.extreme_threshold as et
# %%
class event_extreme:
    """
    A class object to extract positive and negative extreme events from a time series.
    
    """

    def __init__(self, data, column_name='pc'):
        """
        Parameters
        ----------
        data : pandas.Series
            A pandas series with a datetime index.
        """
        self.data = data
        self.positive_events = None
        self.negative_events = None
        self.thresholds = None
        self.column_name = column_name

        # Check if the data is a pandas dataframe with time in column 0
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas dataframe object.")
        if 'time' not in data.columns:
            raise ValueError("Data must have a 'time' column.")
        if column_name not in data.columns:
            raise ValueError(f"Data must have a '{column_name}' column.")
