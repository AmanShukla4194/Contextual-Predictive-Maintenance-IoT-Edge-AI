import os
import json
from config import MODELS_DIR


def load_all_metadata():
    if not os.path.exists(MODELS_DIR):
        print("No saved models found.")
        return []
    entries = []
    for f in os.listdir(MODELS_DIR):
        if f.endswith("_meta.json"):
            with open(os.path.join(MODELS_DIR, f)) as fp:
                entries.append(json.load(fp))
    return sorted(entries, key=lambda x: x["version"])


def compare_models():
    entries = load_all_metadata()
    if not entries:
        return

    print(f"{'Version':<10} {'Macro F1':<12} {'Features':<10}")
    print("-" * 35)
    best = max(entries, key=lambda x: x["macro_f1"])
    for e in entries:
        marker = " <-- best" if e["version"] == best["version"] else ""
        print(f"v{e['version']:<9} {e['macro_f1']:<12.4f} {e['feature_count']:<10}{marker}")

    print(f"\nBest model: v{best['version']} with Macro F1 = {best['macro_f1']:.4f}")
    return best


def get_best_version():
    entries = load_all_metadata()
    if not entries:
        return None
    return max(entries, key=lambda x: x["macro_f1"])["version"]
