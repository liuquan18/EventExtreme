# %%
import pandas as pd
from scipy import ndimage


# %%
def extract_pos_extremes(df, column="residual"):
    """
    extract exsecutively above zero events
    """
    # apply ndimage.median_filter to remove the single day anomaly data (with one day tolerance)
    df.loc[:, column] = ndimage.median_filter(df[column], size=3)
    # A grouper that increaments every time a non-positive value is encountered
    Grouper_pos = df.groupby(df.time.dt.year)[column].transform(
        lambda x: x.lt(0).cumsum()
    )

    # groupby the year and the grouper
    G = df[df[column] > 0].groupby([df.time.dt.year, Grouper_pos])

    # Get the statistics of the group
    Events = G.agg(
        extreme_start_time=pd.NamedAgg(column="time", aggfunc="min"),
        extreme_end_time=pd.NamedAgg(column="time", aggfunc="max"),
        sum=pd.NamedAgg(column=column, aggfunc="sum"),
        mean=pd.NamedAgg(column=column, aggfunc="mean"),
        max=pd.NamedAgg(column=column, aggfunc="max"),
        min=pd.NamedAgg(
            column=column, aggfunc="min"
        ),  # add mean to make sure the data are all positive
    ).reset_index()
    Events["extreme_duration"] = (
        Events["extreme_end_time"] - Events["extreme_start_time"]
    ).dt.days + 1

    Events = Events[
        [
            "extreme_start_time",
            "extreme_end_time",
            "extreme_duration",
            "sum",
            "mean",
            "max",
            "min",
        ]
    ]
    return Events


# %%
def extract_neg_extremes(df, column="residual"):
    """
    extract exsecutively below zero events
    """
    # apply ndimage.median_filter to remove the single day anomaly data (with one day tolerance)
    df[column] = ndimage.median_filter(df[column], size=3)

    # A grouper that increaments every time a non-positive value is encountered
    Grouper_neg = df.groupby(df.time.dt.year)[column].transform(
        lambda x: x.gt(0).cumsum()
    )

    # groupby the year and the grouper
    G = df[df[column] < 0].groupby([df.time.dt.year, Grouper_neg])

    # Get the statistics of the group
    Events = G.agg(
        extreme_start_time=pd.NamedAgg(column="time", aggfunc="min"),
        extreme_end_time=pd.NamedAgg(column="time", aggfunc="max"),
        sum=pd.NamedAgg(column=column, aggfunc="sum"),
        mean=pd.NamedAgg(column=column, aggfunc="mean"),
        max=pd.NamedAgg(column=column, aggfunc="max"),
        min=pd.NamedAgg(
            column=column, aggfunc="min"
        ),  # add mean to make sure the data are all positive
    ).reset_index()
    Events["extreme_duration"] = (
        Events["extreme_end_time"] - Events["extreme_start_time"]
    ).dt.days + 1

    Events = Events[
        [
            "extreme_start_time",
            "extreme_end_time",
            "extreme_duration",
            "sum",
            "mean",
            "max",
            "min",
        ]
    ]
    return Events


# %%
def find_sign_times(extremes, signs, independent_dim=None, combine=False):
    """
    Find the sign_start_time and sign_end_time for each extreme event.

    Parameters:
    extremes (pd.DataFrame): The DataFrame containing the extreme events.
    signs (pd.DataFrame): The DataFrame containing the sign events.
    combine (bool): If True, combine the events with the same sign_start_time and sign_end_time.

    Returns:
    pd.DataFrame: The DataFrame containing the extreme events with sign_start_time and sign_end_time.
    """

    # select rows of signs, where the sign event is within the extreme event
    new_extremes = []
    for i, row in extremes.iterrows():
        if independent_dim is None:
            sign_i = signs[
                (signs["extreme_start_time"] <= row["extreme_start_time"])
                & (signs["extreme_end_time"] >= row["extreme_end_time"])
            ]
        else:
            sign_i = signs[
                (signs["extreme_start_time"] <= row["extreme_start_time"])
                & (signs["extreme_end_time"] >= row["extreme_end_time"])
                & (signs[independent_dim] == row[independent_dim])
            ]
        if not sign_i.empty:

            row["sign_start_time"] = sign_i["extreme_start_time"].values[0]
            row["sign_end_time"] = sign_i["extreme_end_time"].values[0]
            new_extremes.append(row)

    new_extremes = pd.DataFrame(new_extremes)

    # convert the columns to datetime
    date_time_columns = [
        "extreme_start_time",
        "extreme_end_time",
        "sign_start_time",
        "sign_end_time",
    ]
    for col in date_time_columns:
        new_extremes[col] = pd.to_datetime(new_extremes[col])

    if combine:
        # find duplicated rows on 'sign_start_time' and 'sign_end_time', delete first one, and replace the 'start_time' with
        # smallest 'start_time' and 'end_time' with largest 'end_time'
        # group by 'sign_start_time' and 'sign_end_time'
        new_extremes = new_extremes.groupby(["sign_start_time", "sign_end_time"])[
            new_extremes.columns
        ].apply(
            lambda x: x.assign(
                extreme_start_time=x["extreme_start_time"].min(),
                extreme_end_time=x["extreme_end_time"].max(),
                extreme_duration=(
                    x["extreme_end_time"].max() - x["extreme_start_time"].min()
                ).days
                + 1,
            )
        )
        new_extremes = new_extremes.reset_index(drop=True)
        new_extremes["sign_duration"] = (
            new_extremes["sign_end_time"] - new_extremes["sign_start_time"]
        ).dt.days + 1

        new_extremes = new_extremes.drop_duplicates(
            subset=["sign_start_time", "sign_end_time"], ignore_index=True
        )

    return new_extremes
