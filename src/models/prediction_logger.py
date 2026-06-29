import os
import csv
from datetime import datetime
from config import REPORTS_DIR


LOG_FILE = os.path.join(REPORTS_DIR, "prediction_log.csv")
HEADERS = ["timestamp", "machineID", "predicted_label", "confidence", "top2_label", "top2_confidence"]


def _ensure_log():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS)
            writer.writeheader()


def log_prediction(machine_id, predicted_label, proba, label_map):
    _ensure_log()
    sorted_idx = sorted(range(len(proba)), key=lambda i: proba[i], reverse=True)
    top1 = label_map.get(sorted_idx[0], str(sorted_idx[0]))
    top2 = label_map.get(sorted_idx[1], str(sorted_idx[1])) if len(sorted_idx) > 1 else ""

    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "machineID": machine_id,
        "predicted_label": top1,
        "confidence": round(float(proba[sorted_idx[0]]), 4),
        "top2_label": top2,
        "top2_confidence": round(float(proba[sorted_idx[1]]), 4) if len(sorted_idx) > 1 else 0.0,
    }
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        writer.writerow(row)


def read_log():
    _ensure_log()
    rows = []
    with open(LOG_FILE, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows
