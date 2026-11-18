import os
import copy
import inspect
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from Models.CNN import SimpleCNN
from Models.ModifiedCNN import ModifiedCNN
from Models.ModelLast import MiniViT
from Utils.getData import MpoxDataset
from Utils.Preprocessing import get_transforms
from Utils.metrics import save_learning_curve


def find_folds_parent(base_dir, keyword):
    """Mencari folder induk yang di dalamnya terdapat sub-folder berisi folder 'Train'."""
    for root, dirs, _ in os.walk(base_dir):
        if "Train" in dirs and keyword.lower() in root.lower():
            return os.path.dirname(root)
    raise FileNotFoundError(f"Struktur folder Train dengan kata kunci '{keyword}' tidak ditemukan di {base_dir}")


if os.path.exists("/kaggle/input"):
    ORIG_FOLDS_DIR = find_folds_parent("/kaggle/input", "Original")
    AUG_FOLDS_DIR  = find_folds_parent("/kaggle/input", "Augmented")
else:
    ORIG_FOLDS_DIR = find_folds_parent("Dataset", "Original")
    AUG_FOLDS_DIR  = find_folds_parent("Dataset", "Augmented")

ORIG_FOLDS = sorted([d for d in os.listdir(ORIG_FOLDS_DIR) if os.path.isdir(os.path.join(ORIG_FOLDS_DIR, d))])[:5]
AUG_FOLDS  = sorted([d for d in os.listdir(AUG_FOLDS_DIR) if os.path.isdir(os.path.join(AUG_FOLDS_DIR, d))])[:5]

print("ORIG_FOLDS_DIR:", ORIG_FOLDS_DIR, "->", ORIG_FOLDS)
print("AUG_FOLDS_DIR :", AUG_FOLDS_DIR, "->", AUG_FOLDS)

BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(model_cls, num_classes):
    """Mendeteksi nama parameter __init__ model secara otomatis (input_c/in_c, num_classes/output_c/dll)."""
    sig = inspect.signature(model_cls.__init__)
    params = list(sig.parameters.keys())[1:]  # Lewati 'self'
    kwargs = {}
    for p in params:
        if p in ("input_c", "in_c", "in_channels"):
            kwargs[p] = 3
        elif p in ("num_classes", "output_c", "out_classes", "n_classes", "classes"):
            kwargs[p] = num_classes
    return model_cls(**kwargs).to(DEVICE)


def train_model_5fold(model_name, model_cls, class_names, epochs=25):
    img_size = getattr(model_cls, "input_size", 100)
    print(f"\n{'='*60}")
    print(f"MEMULAI TRAINING 5-FOLD: {model_name} (Input: {img_size}x{img_size} | Epochs: {epochs})")
    print(f"{'='*60}")

    train_tf, eval_tf = get_transforms(img_size)
    num_classes = len(class_names)

    for fold_idx in range(5):
        ckpt_path = f"checkpoints/{model_name}_fold{fold_idx+1}.pth"
        if os.path.exists(ckpt_path):
            print(f"\n--- [{model_name}] Fold {fold_idx+1}/5 SUDAH ADA ({ckpt_path}), dilewati ---")
            continue

        orig_fold = ORIG_FOLDS[fold_idx]
        aug_fold  = AUG_FOLDS[fold_idx]
        print(f"\n--- [{model_name}] Fold {fold_idx+1}/5 ({orig_fold} + {aug_fold}) ---")

        train_dirs = [
            os.path.join(ORIG_FOLDS_DIR, orig_fold, "Train"),
            os.path.join(AUG_FOLDS_DIR, aug_fold, "Train")
        ]
        valid_dirs = [os.path.join(ORIG_FOLDS_DIR, orig_fold, "Valid")]

        train_loader = DataLoader(
            MpoxDataset(train_dirs, class_names, train_tf),
            batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
        )
        valid_loader = DataLoader(
            MpoxDataset(valid_dirs, class_names, eval_tf),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        model = build_model(model_cls, num_classes)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        for epoch in range(epochs):
            model.train()
            tr_loss, tr_corr, tr_tot = 0.0, 0, 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                out = model(imgs)
                loss = criterion(out, labels)
                loss.backward()
                optimizer.step()

                tr_loss += loss.item() * imgs.size(0)
                tr_corr += (out.argmax(1) == labels).sum().item()
                tr_tot += labels.size(0)

            model.eval()
            val_loss, val_corr, val_tot = 0.0, 0, 0
            with torch.no_grad():
                for imgs, labels in valid_loader:
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    out = model(imgs)
                    loss = criterion(out, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_corr += (out.argmax(1) == labels).sum().item()
                    val_tot += labels.size(0)

            scheduler.step()
            ep_tr_loss, ep_val_loss = tr_loss / tr_tot, val_loss / val_tot
            ep_tr_acc, ep_val_acc = tr_corr / tr_tot, val_corr / val_tot

            history["train_loss"].append(ep_tr_loss)
            history["val_loss"].append(ep_val_loss)
            history["train_acc"].append(ep_tr_acc)
            history["val_acc"].append(ep_val_acc)

            if ep_val_loss < best_val_loss:
                best_val_loss = ep_val_loss
                best_weights = copy.deepcopy(model.state_dict())

            log_step = 10 if epochs >= 100 else 5
            if (epoch + 1) % log_step == 0 or epoch == 0:
                print(f"Epoch [{epoch+1:03d}/{epochs}] | "
                      f"Train Loss: {ep_tr_loss:.4f}, Acc: {ep_tr_acc:.4f} | "
                      f"Val Loss: {ep_val_loss:.4f}, Acc: {ep_val_acc:.4f}")

        torch.save(best_weights, ckpt_path)

        if fold_idx == 0:
            save_learning_curve(
                history,
                title=f"Learning Curve - {model_name} (Fold 1, {epochs} Epochs)",
                save_path=f"results/learning_curve_{model_name}_fold1.png"
            )

def main():
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    class_names = sorted(os.listdir(os.path.join(ORIG_FOLDS_DIR, ORIG_FOLDS[0], "Train")))
    print("Kelas Terdeteksi (6 Kelas):", class_names)

    # Format: (Nama Model, Kelas Model, Jumlah Epoch)
    models_to_train = [
        ("SimpleCNN", SimpleCNN, 100),
        ("ModifiedCNN", ModifiedCNN, 25),
        ("MiniViT", MiniViT, 25)
    ]

    for name, cls, ep in models_to_train:
        train_model_5fold(name, cls, class_names, epochs=ep)


if __name__ == "__main__":
    main()
