import pandas as pd


def feature_summary(df, feature_cols):
    stats = df[feature_cols].describe().T
    stats["missing"] = df[feature_cols].isnull().sum()
    stats["missing_pct"] = (stats["missing"] / len(df) * 100).round(2)
    return stats


def top_variable_features(df, feature_cols, top_n=20):
    variances = df[feature_cols].var().sort_values(ascending=False)
    return variances.head(top_n)


def label_balance(df, label_col="label"):
    counts = df[label_col].value_counts()
    pct = (counts / len(df) * 100).round(2)
    return pd.DataFrame({"count": counts, "percent": pct})
