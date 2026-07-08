import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(cm, class_names, title="Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_feature_importance(feature_names, importance_values, top_n=20, title="Feature Importance"):
    sorted_idx = np.argsort(importance_values)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh([feature_names[i] for i in sorted_idx],
            [importance_values[i] for i in sorted_idx], color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    return fig


def plot_label_distribution(label_counts, title="Class Distribution"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(label_counts.index.astype(str), label_counts.values, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    plt.tight_layout()
    return fig
