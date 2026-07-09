import os
import joblib
from config import MODELS_DIR, SENSOR_COLS
from src.data.loader import load_all
from src.data.pipeline import build_base_dataset
from src.data.cleaner import fill_missing_sensors, remove_outliers_iqr
from src.features.feature_pipeline import build_features
from src.utils.helpers import encode_labels, split_by_time, get_feature_cols
from src.features.feature_selector import remove_low_variance, correlation_filter
from src.models.trainer import cross_validate, train_lgbm


def run_training():
    print("Loading raw data...")
    data = load_all()

    print("Building base dataset...")
    df = build_base_dataset()
    df = fill_missing_sensors(df, SENSOR_COLS)
    df = remove_outliers_iqr(df, SENSOR_COLS)

    print("Engineering features...")
    df = build_features(df, data["maintenance"], data["errors"])
    df, label_map = encode_labels(df)

    train_df, test_df = split_by_time(df)
    feature_cols = get_feature_cols(train_df)
    feature_cols = remove_low_variance(train_df, feature_cols)
    feature_cols = correlation_filter(train_df, feature_cols)

    X_train = train_df[feature_cols].values
    y_train = train_df["label_enc"].values

    print("Running cross-validation...")
    cross_validate(X_train, y_train)

    print("Training final model on full training set...")
    model = train_lgbm(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(
        {"model": model, "features": feature_cols, "label_map": label_map},
        os.path.join(MODELS_DIR, "lgbm_pdm.pkl"),
    )
    print("Model saved.")
    return model, feature_cols, label_map


if __name__ == "__main__":
    run_training()
