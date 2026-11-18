import os
import inspect
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from Models.CNN import SimpleCNN
from Models.ModifiedCNN import ModifiedCNN
from Models.ModelLast import MiniViT
from Utils.getData import MpoxDataset
from Utils.Preprocessing import get_transforms
from Utils.metrics import compute_metrics, save_confusion_matrix


def find_folds_parent(base_dir, keyword):
    """Mencari folder induk yang di dalamnya terdapat sub-folder berisi folder 'Train'."""
    for root, dirs, _ in os.walk(base_dir):
        if "Train" in dirs and keyword.lower() in root.lower():
            return os.path.dirname(root)
    raise FileNotFoundError(f"Struktur folder Train dengan kata kunci '{keyword}' tidak ditemukan di {base_dir}")


if os.path.exists("/kaggle/input"):
    ORIG_FOLDS_DIR = find_folds_parent("/kaggle/input", "Original")
else:
    ORIG_FOLDS_DIR = find_folds_parent("Dataset", "Original")

ORIG_FOLDS = sorted([d for d in os.listdir(ORIG_FOLDS_DIR) if os.path.isdir(os.path.join(ORIG_FOLDS_DIR, d))])[:5]

BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_cls, num_classes):
    """Mendeteksi nama parameter __init__ model secara otomatis."""
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.keys())[1:]
    kwargs = {}
    for p in params:
        if p in ("input_c", "in_c", "in_channels"):
            kwargs[p] = 3
        elif p in ("num_classes", "output_c", "out_classes", "n_classes", "classes"):
            kwargs[p] = num_classes
    return model_cls(**kwargs).to(DEVICE)


def evaluate_model_5fold(model_name, model_cls, class_names):
    img_size = getattr(model_cls, "input_size", 100)
    print(f"\n{'='*60}")
    print(f"EVALUASI TEST SET 5-FOLD: {model_name} (Input Size: {img_size}x{img_size})")
    print(f"{'='*60}")

    _, eval_tf = get_transforms(img_size)
    num_classes = len(class_names)

    fold_records = []
    total_cm = np.zeros((num_classes, num_classes), dtype=int)

    for fold_idx in range(5):
        orig_fold = ORIG_FOLDS[fold_idx]
        test_dirs = [os.path.join(ORIG_FOLDS_DIR, orig_fold, "Test")]
        test_loader = DataLoader(
            MpoxDataset(test_dirs, class_names, eval_tf),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2
        )

        model = build_model(model_cls, num_classes)

        ckpt_path = f"checkpoints/{model_name}_fold{fold_idx+1}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        model.eval()

        y_true, y_pred, y_prob = [], [], []
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(DEVICE)
                probs = torch.softmax(model(imgs), dim=1)
                preds = probs.argmax(1)

                y_true.extend(labels.numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_prob = np.array(y_prob)

        scores, cm = compute_metrics(y_true, y_pred, y_prob, num_classes=num_classes)
        total_cm += cm

        row = {"Model": model_name, "Fold": f"Fold {fold_idx+1}", **scores}
        fold_records.append(row)
        print(f"Fold {fold_idx+1} -> Acc: {scores['Accuracy']:.4f} | Prec: {scores['Precision']:.4f} | "
              f"Rec: {scores['Recall']:.4f} | F1: {scores['F1_Score']:.4f} | ROC-AUC: {scores['ROC_AUC']:.4f}")

    df_model = pd.DataFrame(fold_records)
    mean_summary = {
        "Model": model_name,
        "Fold": "Rata-Rata 5-Fold",
        "Accuracy": df_model["Accuracy"].mean(),
        "Precision": df_model["Precision"].mean(),
        "Recall": df_model["Recall"].mean(),
        "F1_Score": df_model["F1_Score"].mean(),
        "ROC_AUC": df_model["ROC_AUC"].mean()
    }
    df_model = pd.concat([df_model, pd.DataFrame([mean_summary])], ignore_index=True)
    df_model.to_csv(f"results/evaluasi_{model_name}_5fold.csv", index=False)

    save_confusion_matrix(
        total_cm,
        class_names,
        title=f"Confusion Matrix 5-Fold Test Set - {model_name}",
        save_path=f"results/confusion_matrix_{model_name}_5fold.png"
    )
    return df_model, mean_summary


def main():
    os.makedirs("results", exist_ok=True)
    class_names = sorted(os.listdir(os.path.join(ORIG_FOLDS_DIR, ORIG_FOLDS[0], "Train")))

    models_to_eval = [
        ("SimpleCNN", SimpleCNN),
        ("ModifiedCNN", ModifiedCNN),
        ("MiniViT", MiniViT)
    ]

    comparison_rows = []
    for name, cls in models_to_eval:
        _, mean_summary = evaluate_model_5fold(name, cls, class_names)
        comparison_rows.append(mean_summary)

    df_comparison = pd.DataFrame(comparison_rows)
    print("\n" + "="*80)
    print("PERBANDINGAN RATA-RATA 5-FOLD CROSS VALIDATION (MODEL 1 vs MODEL 2 vs MODEL 3)")
    print("="*80)
    print(df_comparison.to_string(index=False))
    df_comparison.to_csv("results/perbandingan_model_5fold.csv", index=False)


if __name__ == "__main__":
    main()
