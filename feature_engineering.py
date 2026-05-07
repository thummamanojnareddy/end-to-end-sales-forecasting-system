import pandas as pd
import numpy as np

def create_features(df):

    df = df.sort_values(['state', 'date'])

    df['lag_1'] = df.groupby('state')['sales'].shift(1)
    df['lag_7'] = df.groupby('state')['sales'].shift(7)
    df['lag_30'] = df.groupby('state')['sales'].shift(30)

    df['rolling_mean_7'] = (
        df.groupby('state')['sales']
        .transform(lambda x: x.rolling(7).mean())
    )

    df['rolling_std_7'] = (
        df.groupby('state')['sales']
        .transform(lambda x: x.rolling(7).std())
    )

    df['month'] = df['date'].dt.month
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['quarter'] = df['date'].dt.quarter
    df['year'] = df['date'].dt.year

    df['trend'] = np.arange(len(df))

    holidays = [
        '2019-12-25',
        '2020-12-25',
        '2021-12-25'
    ]

    df['holiday_flag'] = (
        df['date'].astype(str).isin(holidays)
    ).astype(int)

    df = df.dropna()

    return df
