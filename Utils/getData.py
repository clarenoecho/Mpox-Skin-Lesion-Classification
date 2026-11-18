import os
from PIL import Image
from torch.utils.data import Dataset

class MpoxDataset(Dataset):
    """Dataset loader untuk Mpox Skin Lesion Dataset v2.0 (6 kelas per fold)."""
    def __init__(self, folder_paths, class_names, transform=None):
        self.samples = []
        self.transform = transform
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}

        for folder in folder_paths:
            if not os.path.exists(folder):
                continue
            for cls_name in class_names:
                cls_dir = os.path.join(folder, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                label = self.class_to_idx[cls_name]
                for img_name in os.listdir(cls_dir):
                    if img_name.lower().endswith((".png", ".jpg", ".jpeg")):
                        self.samples.append((os.path.join(cls_dir, img_name), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
