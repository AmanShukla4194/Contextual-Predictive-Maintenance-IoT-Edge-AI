import pandas as pd
from config import RANDOM_STATE


def stratified_sample(df, label_col, fraction=0.1):
    return (
        df.groupby(label_col, group_keys=False)
        .apply(lambda x: x.sample(frac=fraction, random_state=RANDOM_STATE))
        .reset_index(drop=True)
    )


def temporal_train_test_split(df, datetime_col="datetime", test_fraction=0.2):
    df = df.sort_values(datetime_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def class_counts(df, label_col):
    counts = df[label_col].value_counts().sort_index()
    print("Class distribution:")
    for cls, cnt in counts.items():
        print(f"  {cls}: {cnt} ({cnt / len(df) * 100:.2f}%)")
    return counts
