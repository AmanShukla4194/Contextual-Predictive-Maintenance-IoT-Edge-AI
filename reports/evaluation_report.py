import os
import json
from config import REPORTS_DIR


def load_evaluation_results(path=None):
    if path is None:
        path = os.path.join(REPORTS_DIR, "evaluation_results.txt")
    with open(path, "r") as f:
        return f.read()


def parse_macro_f1(report_text):
    for line in report_text.splitlines():
        if line.startswith("Macro F1:"):
            return float(line.split(":")[1].strip())
    return None


def print_summary(report_text):
    f1 = parse_macro_f1(report_text)
    print("=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    if f1 is not None:
        print(f"Macro F1 Score : {f1:.4f}")
    print()
    print(report_text)


def save_json_summary(report_text, out_path=None):
    if out_path is None:
        out_path = os.path.join(REPORTS_DIR, "evaluation_summary.json")
    f1 = parse_macro_f1(report_text)
    summary = {"macro_f1": f1, "report": report_text}
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"JSON summary saved to {out_path}")


if __name__ == "__main__":
    report = load_evaluation_results()
    print_summary(report)
    save_json_summary(report)
