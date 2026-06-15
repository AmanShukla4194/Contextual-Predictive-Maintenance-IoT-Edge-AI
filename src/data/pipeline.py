from src.data.loader import load_all
from src.data.merger import merge_errors_to_telemetry, merge_maintenance_to_telemetry
from src.data.preprocessor import label_failures, merge_machine_info


def build_base_dataset(data_dir=None):
    kwargs = {"data_dir": data_dir} if data_dir else {}
    data = load_all(**kwargs)

    df = data["telemetry"]
    df = merge_errors_to_telemetry(df, data["errors"])
    df = merge_maintenance_to_telemetry(df, data["maintenance"])
    df = label_failures(df, data["failures"])
    df = merge_machine_info(df, data["machines"])

    return df