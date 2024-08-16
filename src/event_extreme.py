# %%
import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# %%
import src.extreme_extract as ee
import src.extreme_threshold as et

# %%
import importlib

importlib.reload(ee)
importlib.reload(et)


# %%
class EventExtreme:
    """
    A class object to extract positive and negative extreme events from a time series.

    """

    def __init__(self, data, column_name="pc", threshold_std=1.5, independent_dim=None):
        """
        Parameters
        ----------
        data : pandas.Series
            A pandas DataFrame object with a 'time' column and a column to be used for threshold calculation.

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
        self.threshold_std = (
            threshold_std  # the threshold as unit of standard deviation
        )
        self.column_name = column_name

        self.pos_thr_dayofyear = None  # calculate the absolute value of threshold at each day-of-year with 7 day-window
        self.neg_thr_dayofyear = None

        self.positive_events = None
        self.negative_events = None

        self.independent_dim = independent_dim

        # Check if the data is a pandas dataframe with time in one of the columns
        if not isinstance(self.data, pd.DataFrame):
            raise ValueError("Data must be a pandas dataframe object.")
        if "time" not in self.data.columns:
            raise ValueError("Data must have a 'time' column.")
        if self.column_name not in self.data.columns:
            raise ValueError(f"Data must have a '{self.column_name}' column.")

        # check the independent_dim
        self.examine_independent_dim()

        logging.info("remove leap year 29th February")
        # Convert 'time' column to datetime
        self.data.time = pd.to_datetime(self.data.time)

        # Check for any conversion errors
        if self.data["time"].isnull().any():
            logging.warning("There were some errors in converting 'time' to datetime")

        # Remove leap year 29th February
        self.data = self.data[
            ~((self.data["time"].dt.month == 2) & (self.data["time"].dt.day == 29))
        ]

    @property
    def extract_positive_extremes(self):
        """
        propoerty function to extract positive extreme events.
        """
        if self.independent_dim is None:
            self.positive_events = self.extract_extremes_single(extreme_type="pos")
        else:
            self.positive_events = self.extract_extremes_multi(extreme_type="pos")

        logging.info("Positive extreme events are extracted.")
        return self.positive_events

    @property
    def extract_negative_extremes(self):
        """
        propoerty function to extract negative extreme events.
        """
        if self.independent_dim is None:
            self.negative_events = self.extract_extremes_single(extreme_type="neg")
        else:
            self.negative_events = self.extract_extremes_multi(extreme_type="neg")

        logging.info("Negative extreme events are extracted.")
        return self.negative_events

    def examine_independent_dim(self):
        # if there are other dimensions apart from 'time' and 'column_name'
        if (len(self.data.columns) < 3) and (self.independent_dim is None):
            logging.info(
                "single time series data detected. No independent dimension is set."
            )

        elif (len(self.data.columns) >= 3) and (self.independent_dim is None):
            # update the independent_dim with column that is not 'time' and 'column_name'
            self.independent_dim = [
                col
                for col in self.data.columns
                if col not in ["time", self.column_name]
            ][0]
            logging.info(f"Independent dimension is set to {self.independent_dim}")

        else:
            raise ValueError(
                "independent dimension is wrongly set. Please check the data"
            )

    def calculate_threshold_single(self, extreme_type: int = "pos") -> pd.DataFrame:
        """
        Calculate the threshold for positive or negative extreme events.
        The threshold is calculated for each day-of-year with a 7-day window.
        This threshold can be replaced by a user-defined threshold (with data-of-year as one of the columns).

        Parameters
        ----------
        extreme_type: str
            The type of extreme events to calculate the threshold. Default is 'pos' for positive extreme events.
            The other option is 'neg' for negative extreme events.

        Returns
        -------
        thr_dayofyear: pandas.DataFrame
            A pandas DataFrame with self.independent_dim (if appliable), 'time' and 'threshold' columns
        """

        data_window = et.construct_window(
            self.data, column_name=self.column_name, window=7
        )

        if extreme_type == "pos":
            thr_dayofyear = et.threshold(
                data_window, column_name=self.column_name, extreme_type="pos"
            )
        elif extreme_type == "neg":

            thr_dayofyear = et.threshold(
                data_window, column_name=self.column_name, extreme_type="neg"
            )

        return thr_dayofyear

    def calculate_threshold_multi(
        self, independent_dim, extreme_type: int = "pos"
    ) -> pd.DataFrame:
        """
        calculate the threshold idividually for each value of independent_dim.
        """
        data_window = self.data.groupby(self.independent_dim)[
            ["time", self.column_name]
        ].apply(et.construct_window, column_name=self.column_name, window=7)
        data_window = data_window.droplevel(-1).reset_index()

        if extreme_type == "pos":
            thr_dayofyear = data_window.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(et.threshold, column_name=self.column_name, extreme_type="pos")

        elif extreme_type == "neg":
            thr_dayofyear = data_window.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(et.threshold, column_name=self.column_name, extreme_type="neg")

        return thr_dayofyear

    def extract_extremes_single(self, extreme_type="pos"):
        """
        extract extreme events based on the calculated threshold.
        The extreme events have 'extreme_start_time' and 'extreme_end_time' columns indicating the start and end time of the values exceeding the threshold.
        'sign_start_time' and 'sign_end_time' columns are also included to show the start and end time that when the event strats to grow and end.


            |                 :       *   *         :
            |                 :   *             *   :
        1.5 |.................*.....................*.............................
            |              *  :                     :    *
            |           *     :                     :         *
          0 |_______*_________:_____________________:______________*______________
                  *           :                     :                 *
        sign_start_time   extrme_start_time    extreme_end_time      sign_end_time


         Parameters
        ----------
        extreme_type: str
            The type of extreme events to extract. Default is 'pos' for positive extreme events.
            The other option is 'neg' for negative extreme events.
        """
        if extreme_type == "pos":

            # extreme_strat_time and extreme_end_time are calculated after removing the threshold from original data
            pos_thr_dayofyear = self.calculate_threshold_single(extreme_type="pos")
            data_residual = et.subtract_threshold(
                self.data, threshold=pos_thr_dayofyear, column_name=self.column_name
            )

            # extract positive 'extreme' events based on 'residual' column. see source code for more details
            pos_extreme_events = ee.extract_pos_extremes(
                data_residual, column="residual"
            )

            # extract positive 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
            pos_sign_events = ee.extract_pos_extremes(
                self.data, column=self.column_name
            )

            # find the corresponding sign-time for pos_extreme event
            events = ee.find_sign_times(pos_extreme_events, pos_sign_events)

        elif extreme_type == "neg":

            # extreme_strat_time and extreme_end_time are calculated after removing the threshold from original data
            neg_thr_dayofyear = self.calculate_threshold_single(extreme_type="neg")
            data_residual = et.subtract_threshold(
                self.data, threshold=neg_thr_dayofyear, column_name=self.column_name
            )

            # extract negative 'extreme' events based on 'residual' column. see source code for more details
            neg_extreme_events = ee.extract_neg_extremes(
                data_residual, column="residual"
            )

            # extract negative 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
            neg_sign_events = ee.extract_neg_extremes(
                self.data, column=self.column_name
            )

            # find the corresponding sign-time for neg_extreme event
            events = ee.find_sign_times(neg_extreme_events, neg_sign_events)

        return events

    def extract_extremes_multi(self, independent_dim, extreme_type="pos"):
        """
        extract extreme events individually for each value of independent_dim.
        """
        logging.info(
            f"Using groupby('{self.independent_dim}') to extract extreme events didependently for each value of '{self.independent_dim}'"
        )

        if extreme_type == "pos":
            # extreme_strat_time and extreme_end_time are calculated after removing the threshold from original data
            thr_dayofyear = self.calculate_threshold_multi(extreme_type="pos")
            data_residual = self.data.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(
                et.subtract_threshold,
                threshold=thr_dayofyear,
                column_name=self.column_name,
            )
            data_residual = data_residual.droplevel(-1).reset_index()

            # extract positive 'extreme' events based on 'residual' column. see source code for more details
            pos_extreme_events = data_residual.groupby(self.independent_dim)[
                ["time", "residual"]
            ].apply(ee.extract_pos_extremes, column="residual")

            # extract positive 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
            pos_sign_events = self.data.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(ee.extract_pos_extremes, column=self.column_name)

            # find the corresponding sign-time for pos_extreme event
            events = ee.find_sign_times(pos_extreme_events, pos_sign_events)

        elif extreme_type == "neg":
            # extreme_strat_time and extreme_end_time are calculated after removing the threshold from original data
            thr_dayofyear = self.calculate_threshold_multi(extreme_type="neg")
            data_residual = self.data.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(
                et.subtract_threshold,
                threshold=thr_dayofyear,
                column_name=self.column_name,
            )
            data_residual = data_residual.droplevel(-1).reset_index()

            # extract negative 'extreme' events based on 'residual' column. see source code for more details
            neg_extreme_events = data_residual.groupby(self.independent_dim)[
                ["time", "residual"]
            ].apply(ee.extract_neg_extremes, column="residual")

            # extract negative 'sign' events based on column_name. This is for find sign_start_time and sign_end_time
            neg_sign_events = self.data.groupby(self.independent_dim)[
                ["time", self.column_name]
            ].apply(ee.extract_neg_extremes, column=self.column_name)

            # find the corresponding sign-time for neg_extreme event
            events = ee.find_sign_times(neg_extreme_events, neg_sign_events)

        return events
