#%%
import pandas as pd
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

# %%
import src.extreme_extract as ee
import src.extreme_threshold as et

#%%
import importlib
importlib.reload(ee)
importlib.reload(et)
# %%
class EventExtreme_single:
    """
    A class object to extract positive and negative extreme events from a time series.
    
    """

    def __init__(self, data, column_name='pc', threshold_std=1.5, independent_dim = None):
        """
        Parameters
        ----------
        data : pandas.Series
            A pandas series with a datetime index.
        
        column_name: str
            The name of the column to be used in the threshold calculation.

        threshold_std: float
            The threshold value. Default is 1.5 standard deviation.

        independent_dim: str
            Extremes should be extracted independently for each value of this dimension. 
            for example, if the data is 3D with dimensions ['plev','time','pc'], then
            the independent_dim can be 'plev' and the extreme events are extracted independently for each value of 'plev'.
        """
        self.data = data
        self.threshold_std = threshold_std # the threshold as unit of standard deviation
        self.column_name = column_name

        self.pos_abs_thr = None # calculate the absolute value of threshold at each day-of-year with 7 day-window
        self.neg_abs_thr = None

        self.positive_events = None
        self.negative_events = None

        self.independent_dim = independent_dim 

        # Check if the data is a pandas dataframe with time in column 0
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas dataframe object.")
        if 'time' not in data.columns:
            raise ValueError("Data must have a 'time' column.")
        if column_name not in data.columns:
            raise ValueError(f"Data must have a '{column_name}' column.")
        
        # if there are other dimensions apart from 'time' and 'column_name'
        if (len(data.columns) == 3) and (independent_dim is None):
            # update the independent_dim with column that is not 'time' and 'column_name'
            self.independent_dim = [col for col in data.columns if col not in ['time', column_name]][0]
            logging.info(f"Independent dimension is set to {self.independent_dim}")

        if (len(data.columns) > 3):
            raise ValueError("There are more than 3 columns in the data")
        

        logging.info("remove leap year 29th February")

        # Convert 'time' column to datetime
        self.data.time = pd.to_datetime(self.data.time)

        # Check for any conversion errors
        if self.data['time'].isnull().any():
            logging.warning("There were some errors in converting 'time' to datetime")

        # Remove leap year 29th February
        self.data = self.data[~((self.data['time'].dt.month == 2) & (self.data['time'].dt.day == 29))]
    @property
    def extract_positive_extremes(self):
        
        # step1: calculate the positive threshold for each day-of-year with a 7-day window
        if self.pos_abs_thr is None:
            logging.info("positive threshold not provided. Calculating threshold.")

        data_window = et.construct_window(self.data, column_name = self.column_name, window=7)
        self.pos_abs_thr = et.threshold(data_window, column_name = self.column_name, extreme_type = 'pos')

        # step2: subtract the threshold from the data, and construct a new column called 'residual'
        pos_data = et.subtract_threshold(self.data, threshold=self.pos_abs_thr, column_name=self.column_name)
        
        # step3: extract positive 'extreme' events based on 'residual' column. see source code for more details
        pos_extreme = ee.extract_pos_extremes(pos_data, column='residual')

        # step4: extract positive 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
        pos_sign = ee.extract_pos_extremes(self.data, column=self.column_name)

        # step 5: find the corresponding sign-time for pos_extreme event
        self.positive_events = ee.find_sign_times (pos_extreme, pos_sign)


    @property
    def extract_negative_extremes(self):
            
        # step1: calculate the negative threshold for each day-of-year with a 7-day window
        if self.neg_abs_thr is None:
            logging.info("negative threshold not provided. Calculating threshold.")

        data_window = et.construct_window(self.data, column_name = self.column_name, window=7)
        self.neg_abs_thr = et.threshold(data_window, column_name = self.column_name, extreme_type = 'neg')

        # step2: subtract the threshold from the data, and construct a new column called 'residual'
        neg_data = et.subtract_threshold(self.data, threshold=self.neg_abs_thr, column_name=self.column_name)
        
        # step3: extract negative 'extreme' events based on 'residual' column. see source code for more details
        neg_extreme = ee.extract_neg_extremes(neg_data, column='residual')

        # step4: extract negative 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
        neg_sign = ee.extract_neg_extremes(self.data, column=self.column_name)

        # step 5: find the corresponding sign-time for neg_extreme event
        self.negative_events = ee.find_sign_times (neg_extreme, neg_sign)


class EventExtreme_multi:
    """
    A class object to extract positive and negative extreme events from a time series with multiple dimensions.
    
    """

    def __init__(self, data, independent_dim = None, dependent_dim = None, column_name='pc', threshold_std=1.5):
        """
        Parameters
        ----------
        data : pandas.Series
            A pandas series with a datetime index.

        independent_dim: str
            The independent dimension that doesn't used to calculate absolute thresholds.
            extremes events are extracted independently for each value of this dimension. 
            For example, if the data is 3D with dimensions ['plev','time','pc'], 
            then the independent_dim can be 'plev' and the extreme events are extracted independently for each value of 'plev'.

        dependent_dim: str
            The dependent dimension that used to calculate absolute thresholds.
            The absolute thresholds are calculated along this dimension and time dimension (and window dimension).
            For example, if the data is 3D with dimensions ['plev','ens','time','pc'], 
            then the dependent_dim can be 'ens' so that the threshold also claculates on 'ens' along with time.
        """
        self.data = data
        self.threshold_std = threshold_std

        self.column_name = column_name
        self.independent_dim = independent_dim

        self.pos_abs_thr = None # calculate the absolute value of threshold at each day-of-year with 7 day-window
        self.neg_abs_thr = None

        self.positive_events = None
        self.negative_events = None


    # check if 'time', independent_dim and dependent_dim, column_name are in the columns
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a pandas dataframe object.")
        if 'time' not in data.columns:
            raise ValueError("Data must have a 'time' column.")
        if column_name not in data.columns:
            raise ValueError(f"Data must have a '{column_name}' column.")
        if independent_dim is not None and independent_dim not in data.columns:
            raise ValueError(f"Data must have a '{independent_dim}' column.")
        if dependent_dim is not None and dependent_dim not in data.columns:
            raise ValueError(f"Data must have a '{dependent_dim}' column.")
        
        logging.info("remove leap year 29th February")
        # Convert 'time' column to datetime
        self.data.time = pd.to_datetime(self.data.time)
        # Remove leap year 29th February
        self.data = self.data[~((self.data['time'].dt.month == 2) & (self.data['time'].dt.day == 29))]


    @property
    def extract_positive_extremes(self):
            
        # step1: calculate the positive threshold for each day-of-year with a 7-day window
        if self.pos_abs_thr is None:
            logging.info("positive threshold not provided. Calculating threshold.")
    
            # windows can only be constructed for 1D time series data by using 'shift'
            try:
                G_window = self.data.groupby([self.independent_dim, self.dependent_dim])
            except TypeError:
                G_window = self.data.groupby(self.independent_dim)

            data_window = G_window.apply(et.construct_window, column_name = self.column_name, window=7)
            data_window = 

            self.pos_abs_thr = et.threshold(data_window, column_name = self.column_name, extreme_type = 'pos')
    
            # step2: subtract the threshold from the data, and construct a new column called 'residual'
            pos_data = et.subtract_threshold(self.data, threshold=self.pos_abs_thr, column_name=self.column_name)
            
            # step3: extract positive 'extreme' events based on 'residual' column. see source code for more details
            pos_extreme = ee.extract_pos_extremes(pos_data, column='residual')
    
            # step4: extract positive 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
            pos_sign = ee.extract_pos_extremes(self.data, column=self.column_name)
    
            # step 5: find the corresponding sign-time for pos_extreme event
            self.positive_events = ee.find_sign_times (pos_extreme, pos_sign)