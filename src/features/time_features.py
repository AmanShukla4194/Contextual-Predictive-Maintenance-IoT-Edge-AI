import pandas as pd


def add_time_features(df, datetime_col="datetime"):
    df = df.copy()
    dt = pd.to_datetime(df[datetime_col])
    df["hour_of_day"] = dt.dt.hour
    df["day_of_week"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    return df


def drop_time_features(df):
    cols = ["hour_of_day", "day_of_week", "month", "is_weekend", "week_of_year"]
    return df.drop(columns=[c for c in cols if c in df.columns])
