from datetime import datetime

import pandas
import pandas as pd


def index_to_day_date_range(index):
    # get indexes by day and skip minute granularity
    return pd.date_range(index[0], index[-1] + pd.offsets.Day())


def process_dataframe_dtypes(df: pandas.DataFrame, datetime_cols=None, timestamp_cols=None, remove_empty_cols=True):
    """
    :param datetime_cols: columns to convert to datetimes
    :param remove_empty_cols: remove columns that are all nans
    """
    if datetime_cols is None:
        datetime_cols = []
    if timestamp_cols is None:
        timestamp_cols = []

    for col in df.columns:
        if remove_empty_cols and df[col].apply(lambda x: x is None).all():
            # all values are None.
            del df[col]
            continue

        if col in timestamp_cols:
            df[col] = df[col].map(datetime.fromtimestamp)
        elif col in datetime_cols:
            df[col] = pandas.to_datetime(df[col])
        else:
            try:
                # by default try to convert ot numeric
                df[col] = pandas.to_numeric(df[col])
            except:  # noqa  - TODO: check proper exception later
                pass

    return df
