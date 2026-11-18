import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)
from sklearn.preprocessing import label_binarize

def compute_metrics(y_true, y_pred, y_prob, num_classes=6):
    """Menghitung seluruh metrik evaluasi yang diwajibkan pada tugas."""
    cm = confusion_matrix(y_true, y_pred)
    y_bin = label_binarize(y_true, classes=list(range(num_classes)))

    scores = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1_Score": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    }
    return scores, cm


def save_learning_curve(history, title, save_path):
    """Menyimpan grafik kurva Loss dan Accuracy selama training."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss", linestyle="--")
    axes[0].set_title(f"{title} - CrossEntropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history["train_acc"], label="Train Acc")
    axes[1].plot(history["val_acc"], label="Val Acc", linestyle="--")
    axes[1].set_title(f"{title} - Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def save_confusion_matrix(cm, class_names, title, save_path):
    """Menyimpan heatmap akumulasi Confusion Matrix 5-Fold."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel("Prediksi Model")
    plt.ylabel("Label Aktual")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
