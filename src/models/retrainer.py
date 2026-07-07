import pandas as pd
from src.data.preprocessor import merge_machine_info
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.models.trainer import train_lgbm, cross_validate
from src.models.model_registry import save_model_version
from src.data.loader import load_all
from config import RANDOM_STATE


def retrain(new_telemetry_df=None, version=2):
    data = load_all()

    if new_telemetry_df is not None:
        existing = data["telemetry"]
        combined = pd.concat([existing, new_telemetry_df]).drop_duplicates(
            subset=["machineID", "datetime"]
        ).sort_values(["machineID", "datetime"]).reset_index(drop=True)
        data["telemetry"] = combined
        print(f"Retrain dataset size: {len(combined)} rows (added {len(new_telemetry_df)} new rows)")

    from src.data.pipeline import build_base_dataset
    df = build_base_dataset(data)
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)

    train_df, _ = split_by_time(df)
    feature_cols = get_feature_cols(train_df)
    X_train = train_df[feature_cols].values
    y_train = train_df["label_enc"].values

    print("Running cross-validation on new dataset...")
    _, mean_f1, _ = cross_validate(X_train, y_train)

    print("Training final model...")
    model = train_lgbm(X_train, y_train)
    save_model_version(model, feature_cols, label_map, version=version, score=mean_f1)
    print(f"Retraining complete. Saved as version {version}.")
    return model, feature_cols, label_map
