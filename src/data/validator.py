import pandas as pd


def check_missing(df, name):
    missing = df.isnull().sum()
    if missing.any():
        print(f"[{name}] missing values:\n{missing[missing > 0]}")
    else:
        print(f"[{name}] no missing values")


def check_date_range(df, name):
    print(f"[{name}] date range: {df['datetime'].min()} -> {df['datetime'].max()}")


def summarise(datasets):
    for name, df in datasets.items():
        print(f"\n--- {name} ---")
        print(f"  rows: {len(df):,}  cols: {df.shape[1]}")
        if "datetime" in df.columns:
            check_date_range(df, name)
        check_missing(df, name)
