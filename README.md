# Klasifikasi Lesi Kulit Mpox (MSLD v2.0) — Perbandingan 3 Model Deep Learning

Studi kasus Minggu 10 (Pembelajaran Mesin / Deep Learning) — Fakultas Informatika, Telkom University.
Klasifikasi citra kulit ke **6 kelas**: Chickenpox, Cowpox, Healthy, HFMD, Measles, Monkeypox, dibandingkan
dengan 5-fold cross validation pada tiga model.

> ⚠️ **STATUS README:** bagian bertanda `⏳ ISI` harus diisi dari `results/ANALYSIS.md` setelah eksperimen dijalankan.
> Hapus baris ini sebelum dikumpulkan.

## 1. Latar Belakang & Alasan Memakai 3 Model
Penularan Mpox perlu dideteksi sejak dini, sementara tenaga ahli untuk menilai lesi kulit terbatas. Sistem klasifikasi
citra otomatis diharapkan **efisien, presisi, dan akurat**. Kami membandingkan tiga pendekatan yang mewakili tiga
tingkat kompleksitas agar terlihat apa yang sebenarnya meningkatkan kinerja:

| # | Model | Peran dalam perbandingan |
|---|-------|--------------------------|
| 1 | **Simple CNN** (arsitektur materi W9) | baseline |
| 2 | **Modified CNN** | efek modifikasi arsitektur (input, kernel, aktivasi, BatchNorm, Dropout, GAP) |
| 3 | **MiniViT** (Vision Transformer kompak) | keluarga algoritma berbeda (Transformer, salah satu algoritma di materi W9) |

Model 3 tidak termasuk model pada kode Kaggle MSLD v2.0 (sesuai ketentuan "selain yang ada di Kaggle").
## 2. Kajian Teori Singkat
- **CNN**: konvolusi mengekstrak fitur lokal (tepi, tekstur, warna lesi); pooling mengurangi dimensi; lapisan FC memetakan fitur ke kelas.
- **BatchNorm / Dropout / GAP**: menstabilkan training, mengurangi overfitting, dan mengurangi jumlah parameter dibanding Flatten+FC.
- **Vision Transformer**: citra dipotong menjadi patch, tiap patch menjadi token, *self-attention* menghubungkan seluruh patch. Butuh data banyak bila dilatih dari nol.
- **Loss**: CrossEntropy untuk klasifikasi multikelas. **Metrik**: confusion matrix, accuracy, precision, recall, F1 (macro), ROC-AUC (One-vs-Rest, macro).

## 3. Metode
### 3.1 Dataset
[Mpox Skin Lesion Dataset v2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20):
755 citra original dari 541 pasien, 6 kelas (Monkeypox 284, HFMD 161, Healthy 114, Chickenpox 75, Cowpox 66, Measles 55 menurut literatur;
verifikasi dengan `results/ANALYSIS.md`). Tersedia dalam 5 fold (Train/Valid/Test) + citra augmented khusus Train (≈14×).
Hanya dataset ini yang dipakai (tidak ada data eksternal, **bobot dilatih dari nol / tanpa pre-trained**).

### 3.2 Protokol (sesuai ketentuan tugas)
- **Training** = *Original Train* + *Augmented Train* fold ke-k.
- **Validasi** = *Original Valid* (pemilihan epoch terbaik via macro-F1 validasi + early stopping).
- **Testing** = *Original Test* — tidak pernah dipakai untuk training maupun pemilihan model.
- Ulangi untuk fold 1–5, laporkan **rata-rata ± std** (std sampel, ddof=1).
- Loss `CrossEntropyLoss`; optimizer AdamW (wd=0 ⇒ Adam, sama seperti materi); batch 32; maks 40 epoch; cosine LR; seed 42+fold.
- Pipeline dan hiperparameter dasar **sama untuk semua model** agar perbandingan adil. Hanya lr/weight-decay/warmup dan ukuran input mengikuti kebutuhan masing-masing (lihat `Models/__init__.py`).

### 3.3 Preprocessing (dieksplorasi)
| Mode | Langkah |
|------|---------|
| `basic` | resize (100 / 128 px) + normalisasi mean/std |
| `clahe` | resize + CLAHE pada kanal L (ruang LAB) + normalisasi |
| `grayworld` | resize + koreksi warna gray-world + normalisasi |

### 3.4 Arsitektur
**Model 1 — Simple CNN** (identik dengan materi; input 3×100×100): `Conv(3→20,5×5)-ReLU-MaxPool(3)` → `Conv(20→50,5×5)-ReLU-MaxPool(3)` → Flatten (50·9·9) → `FC(→6)`.

**Model 2 — Modified CNN** (input 3×128×128). Perubahan dari Model 1:
| Aspek | Simple CNN | Modified CNN |
|---|---|---|
| Input | 100×100 | 128×128 |
| Blok konvolusi | 2 blok, 1 conv/blok, kernel 5×5 | 3 blok (32-64-128), 2 conv/blok, kernel 3×3 |
| Normalisasi | – | BatchNorm2d |
| Aktivasi | ReLU | LeakyReLU(0.1) |
| Pooling | MaxPool 3×3 | MaxPool 2×2 |
| Head | Flatten → FC | Global Avg Pool → Dropout(0.5) → FC |

**Model 3 — MiniViT** (input 3×128×128): patch 16×16 → 64 token + [CLS], dim 192, 6 layer encoder, 6 head, MLP ratio 2, GELU, pre-norm, dropout 0.1.

> Catatan: materi memakai LogSoftmax + CrossEntropyLoss. Di sini model mengeluarkan logit + CrossEntropyLoss (log-softmax sudah di dalam loss), hasil matematisnya sama.

## 4. Struktur Repository
```
Models/CNN.py        SimpleCNN, ModifiedCNN          (Anggota 1 & 2)
Models/ViT.py        MiniViT                          (Anggota 3)
Models/__init__.py   registry model + hiperparameter
Utils/data_index.py  penelusuran dataset, preprocessing   (Anggota 1)
Utils/getData.py     Dataset & DataLoader PyTorch          (Anggota 1)
Utils/engine.py      loop training/validasi/test, 5-fold
Utils/metrics.py     confusion matrix, acc, precision, recall, F1, ROC-AUC, plot (Anggota 3)
train.py / test.py   CLI latih+uji / uji ulang
analyze.py           rekap semua run -> results/ANALYSIS.md
results/             gambar training, confusion matrix, ROC, metrics.json (tanpa *.pt)
```
Model terlatih (`*.pt`) dan dataset **tidak** di-commit (lihat `.gitignore`).

## 5. Cara Menjalankan (Google Colab / GPU)
```python
!git clone https://github.com/<username>/<repo>.git && cd <repo>
!pip install -q kagglehub opencv-python-headless seaborn tqdm
import kagglehub
DATA = kagglehub.dataset_download("joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20")
print(DATA)
```
```bash
python train.py --model simple_cnn --data_root $DATA --smoke     # uji cepat dulu
bash run_all.sh $DATA                                            # semua eksperimen + rekap
```
Opsi lain: `--preprocess clahe|grayworld`, `--class_weight`, `--no_aug`, `--folds 1 2`, `--epochs N`.

## 6. Hasil
⏳ ISI: tempel tabel "Rata-rata 5 fold" dari `results/ANALYSIS.md` (Soal 6) dan gambar:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Simple CNN | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| Modified CNN | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |
| MiniViT | ⏳ | ⏳ | ⏳ | ⏳ | ⏳ |

Gambar: `results/<run>/fold1/training_curve.png`, `results/<run>/confusion_matrix_pooled.png`, `results/<run>/fold1/roc_curve.png`, `results/comparison_models.png`.

## 7. Analysis and Discussion
⏳ ISI semua jawaban dengan angka dari `results/ANALYSIS.md` (bagian dengan nomor soal yang sama).
1. **Distribusi data per kelas** — ⏳ (Soal 1: gunakan tabel & `class_distribution.png`)
2. **Preprocessing & pengaruhnya** — ⏳ (Soal 2: bandingkan basic vs clahe vs grayworld, dasar pemilihan = F1 validasi)
3. **Apakah model mampu mengklasifikasi 6 kelas dengan baik?** — ⏳ (Soal 3: recall/F1 per kelas)
4. **Kelas yang paling sering salah** — ⏳ (Soal 4: pasangan tertukar terbanyak)
5. **Hasil tiap fold** — ⏳ (Soal 5: tabel per fold)
6. **Rata-rata 5 fold** — ⏳ (Soal 6)
7. **Kelebihan & keterbatasan** — ⏳ (Soal 7: tabel efisiensi; bahas data kecil, imbalance, ViT dari nol, citra web-scraped)

## 8. Kesimpulan
⏳ ISI (paragraf, tanpa pointer): model terbaik, alasan, dan saran pengembangan (pre-trained/transfer learning bila diizinkan, class-weighting, augmentasi online, ensembel, validasi klinis).

## 9. Pembagian Tugas
| Anggota | Tanggung jawab | File |
|---|---|---|
| 1 | Dataset, preprocessing, data loader, split Train/Valid/Test, Simple CNN | `Utils/data_index.py`, `Utils/getData.py`, `Models/CNN.py (SimpleCNN)` |
| 2 | Modified CNN, training & validasi, pencatatan parameter, bandingkan dgn Simple CNN | `Models/CNN.py (ModifiedCNN)`, `Utils/engine.py` |
| 3 | Model 3, evaluasi (confusion matrix, acc, precision, recall, F1, ROC-AUC), visualisasi, rekap 3 model | `Models/ViT.py`, `Utils/metrics.py`, `analyze.py` |

## 10. Referensi
[1] N. Ali et al., "Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)," Kaggle / arXiv:2306.14169, 2023.
[2] A. Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale," ICLR, 2021.
[3] Materi W9 "Deep Learning Menggunakan Python," Fakultas Informatika, Telkom University.
