# %%
import pandas as pd


# %%
def threshold(
    df: pd.DataFrame,
    column_name: str = "pc",
    threshold: int = 1.5,
    extreme_type: str = "pos",
) -> pd.DataFrame:
    """
    Calculate the threshold for the anomaly data across multiple years.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['time', column_name].
    column_name (str): The name of the column to be used in the threshold calculation.
    threshold (float): The threshold value. Default is 1.5 standard deviation.
    type (str): The type of threshold. Default is 'pos'.


    """

    # make the dayofyear the same for both the normal year and leap year
    times = pd.to_datetime(df.time.values)

    # Identify leap years
    is_leap_year = times.is_leap_year

    # Adjust dayofyear for dates from March 1st onward in leap years
    adjusted_dayofyear = times.dayofyear - is_leap_year * (
        (times.month > 2).astype(int)
    )

    # Now, incorporate this adjustment back into your xarray object
    df["adjusted_dayofyear"] = adjusted_dayofyear

    # groupby dayofyear
    G = df.groupby("adjusted_dayofyear")
    # 1.5 standard deviation as the threshold, suppose the mean is already zero (anomaly data)

    if extreme_type == "pos":
        # the standard deviation of the data is calculated from all possible columns
        threshold = 1.5 * G[column_name].std()
    elif extreme_type == "neg":
        threshold = -1.5 * G[column_name].std()
    threshold = pd.DataFrame(threshold)
    threshold = threshold.reset_index()
    threshold.columns = ["dayofyear", "threshold"]

    return threshold


# %%


def construct_window(
    df: pd.DataFrame, column_name: str = "pc", window: int = 7
) -> pd.DataFrame:
    """
    Create a dataframe, with window/2 - 1 days before and after the day as the window.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['time', 'another'].
    column_name (str): The name of the column to be used in the window.
    window (int): The size of the window. Default is 7.

    Returns:
    pd.DataFrame: A dataframe with the constructed window.
    """
    # remove 29.02 if it's a leap year
    df = df[~((df["time"].dt.month == 2) & (df["time"].dt.day == 29))]

    # Set time as index
    df = df.set_index("time")

    # Create a window
    windows = [i for i in range(int(-(window - 1) / 2), int((window - 1) / 2 + 1))]
    df_window = pd.concat(
        [df[column_name].shift(periods=i, freq="1D").rename(i) for i in windows], axis=1
    ).dropna(axis=0, how="any")
    df_stack = df_window.stack().reset_index()
    df_stack.columns = ["time", "window", column_name]

    return df_stack


# %%


def subtract_threshold(
    df: pd.DataFrame, threshold: pd.DataFrame, column_name: str = "pc"
) -> pd.DataFrame:
    """
    Subtract the threshold from the column_name for each day-of-year in df.

    Parameters:
    df (pd.DataFrame): Input dataframe with columns ['time', column_name].
    threshold (pd.DataFrame): Input dataframe with columns ['dayofyear', 'threshold'].
    column_name (str): The name of the column to be used in the threshold calculation.

    Returns:
    pd.DataFrame: Dataframe with the threshold subtracted from the specified column, named as 'residual'.
    """
    # Make the dayofyear the same for both the normal year and leap year
    times = pd.to_datetime(df["time"].values)

    # Identify leap years
    is_leap_year = times.is_leap_year

    # Adjust dayofyear for dates from March 1st onward in leap years
    adjusted_dayofyear = times.dayofyear - is_leap_year * (
        (times.month > 2).astype(int)
    )

    # Add adjusted_dayofyear to the dataframe
    df["adjusted_dayofyear"] = adjusted_dayofyear

    # Ensure the threshold dataframe has a 'dayofyear' column
    if "dayofyear" not in threshold.columns:
        raise ValueError("The threshold dataframe must have a 'dayofyear' column.")

    # Merge the threshold with the df on 'adjusted_dayofyear' and any other columns apart from column_name
    left_on = [col for col in df.columns if (col != column_name) & (col != "time")]
    right_on = [col for col in threshold.columns if col != "threshold"]

    df = pd.merge(df, threshold, left_on=left_on, right_on=right_on, how="left")

    # Subtract the threshold from the specified column
    df["residual"] = df[column_name] - df["threshold"]

    # Drop the 'adjusted_dayofyear' and 'dayofyear' columns
    df = df.drop(columns=["adjusted_dayofyear", "dayofyear"])

    return df
