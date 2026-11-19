# Klasifikasi Lesi Kulit Mpox (MSLD v2.0) — Perbandingan 3 Model Deep Learning

Studi kasus Minggu 10 (Pembelajaran Mesin / Deep Learning) — Fakultas Informatika, Program Studi Sains Data, Telkom University Surabaya.
Klasifikasi citra kulit ke **6 kelas**: Chickenpox, Cowpox, Healthy, HFMD, Measles, dan Monkeypox, dibandingkan dengan **5-fold cross validation** pada tiga model yang dilatih dari nol (*from scratch*).

## 1. Latar Belakang & Alasan Memakai 3 Model
Penularan Mpox perlu dideteksi sejak dini, sementara tenaga ahli untuk menilai lesi kulit terbatas. Sistem klasifikasi citra otomatis diharapkan **efisien, presisi, dan akurat**. Kami membandingkan tiga pendekatan yang mewakili tiga tingkat kompleksitas agar terlihat apa yang sebenarnya meningkatkan kinerja:

| # | Model | Peran dalam Perbandingan |
|---|---|---|
| 1 | **Simple CNN** (`Models/CNN.py`) | Baseline (arsitektur dasar materi kuliah) |
| 2 | **Modified CNN** (`Models/ModifiedCNN.py`) | Efek modifikasi arsitektur (input, kernel, aktivasi, BatchNorm, Dropout, GAP) |
| 3 | **MiniViT** (`Models/ModelLast.py`) | Keluarga algoritma berbeda (*Vision Transformer* kompak dari nol) |

Model 3 tidak termasuk model pada kode Kaggle MSLD v2.0 (sesuai ketentuan eksperimen tanpa *pre-trained weights*).

## 2. Kajian Teori Singkat
- **CNN**: konvolusi mengekstrak fitur lokal (tepi, tekstur, warna lesi); pooling mengurangi dimensi; lapisan FC memetakan fitur ke kelas.
- **BatchNorm / Dropout / GAP**: menstabilkan training, mengurangi *overfitting*, dan mengurangi jumlah parameter dibanding `Flatten + FC`.
- **Vision Transformer**: citra dipotong menjadi *patch*, tiap *patch* menjadi token, *self-attention* menghubungkan seluruh *patch*. Membutuhkan data besar atau pra-pelatihan bila tidak memiliki *inductive bias* spasial lokal.
- **Loss**: `CrossEntropyLoss` untuk klasifikasi multikelas. **Metrik**: *confusion matrix*, *accuracy*, *precision*, *recall*, *F1-score* (*macro*), dan *ROC-AUC* (*One-vs-Rest*, *macro*).

## 3. Metode
### 3.1 Dataset
[Mpox Skin Lesion Dataset v2.0 (MSLD v2.0)](https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20):
**755 citra original** dari **541 pasien** dalam **6 kelas** (Monkeypox 284, HFMD 161, Healthy 114, Chickenpox 75, Cowpox 66, Measles 55). Tersedia dalam 5 *fold* berbasis pasien (`Train`/`Valid`/`Test`) ditambah citra *augmented* khusus `Train` ($\approx 14\times$). Hanya dataset ini yang dipakai (tidak ada data eksternal, **bobot dilatih dari nol / tanpa *pre-trained***).

### 3.2 Protokol Pelatihan & Pengujian
- **Training** = *Original Train* + *Augmented Train* fold ke-$k$.
- **Validasi** = *Original Valid* (pemilihan bobot *epoch* terbaik berdasarkan `val_loss` terendah).
- **Testing** = *Original Test* (total akumulasi **364 sampel uji** lintas 5 *fold*) — tidak pernah dipakai untuk pelatihan maupun pemilihan model.
- Ulangi untuk *fold* 1–5, laporkan **rata-rata ± std** (simpangan baku sampel, $\text{ddof}=1$).
- Loss `CrossEntropyLoss`; optimizer `AdamW` (`lr=0.001`, `weight_decay=0.0001`); `batch_size=32`; penjadwalan `CosineAnnealingLR`; akselerasi *Automatic Mixed Precision* (AMP) pada GPU NVIDIA GeForce RTX; 100 *epoch* untuk `SimpleCNN` serta 25 *epoch* untuk `ModifiedCNN` dan `MiniViT`.

### 3.3 Preprocessing & Augmentasi (`Utils/Preprocessing.py`)
| Tahapan | Implementasi pada Kode |
|---|---|
| `basic` (Semua Split) | Konversi RGB ($C=3$), *resize* ($100\times100$ px untuk `SimpleCNN`; $128\times128$ px untuk `ModifiedCNN` & `MiniViT`), konversi tensor $[0, 1]$, dan normalisasi kanal standar (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`). |
| Augmentasi Khusus `Train` | Penggabungan data *offline* `Augmented Train` + augmentasi *on-the-fly*: `RandomHorizontalFlip(0.5)`, `RandomVerticalFlip(0.3)`, `RandomRotation(15°)`, dan `ColorJitter(brightness=0.1, contrast=0.1)`. |

*(Catatan: Skema koreksi kontras/warna seperti CLAHE dan gray-world tidak diaktifkan dalam implementasi akhir untuk menjaga konsistensi warna alami lesi kulit).*

### 3.4 Arsitektur
**Model 1 — Simple CNN** (`Models/CNN.py`, input $3\times100\times100$): `Conv(3→20, 5×5)-ReLU-MaxPool(3)` → `Conv(20→50, 5×5)-ReLU-MaxPool(3)` → `Flatten` ($50\cdot9\cdot9 = 4.050$) → `FC(→6)`.

**Model 2 — Modified CNN** (`Models/ModifiedCNN.py`, input $3\times128\times128$). Perubahan dari Model 1:
| Aspek | Simple CNN | Modified CNN |
|---|---|---|
| Input | $100\times100$ | $128\times128$ |
| Blok konvolusi | 2 blok, 1 conv/blok, kernel $5\times5$ | 3 blok (32-64-128), 2 conv/blok, kernel $3\times3$ |
| Normalisasi | – | `BatchNorm2d` |
| Aktivasi | `ReLU` | `LeakyReLU(0.1)` |
| Pooling | `MaxPool 3×3` | `MaxPool 2×2` |
| Head | `Flatten` → `FC` | `Global Avg Pool` → `Dropout(0.5)` → `FC` |
| Parameter | 50.876 | 288.230 |

**Model 3 — MiniViT** (`Models/ModelLast.py`, input $3\times128\times128$): patch $16\times16$ → 64 token + `[CLS]`, dim 192, 6 layer encoder, 6 head, MLP ratio 2, `GELU`, *pre-norm*, dropout 0.1 (**1.944.006 parameter**).

## 4. Struktur Repository
```
Mpox-Skin-Lesion-Classification/
├── Models/
│   ├── CNN.py               # Arsitektur SimpleCNN (baseline)
│   ├── ModifiedCNN.py       # Arsitektur ModifiedCNN (model terbaik)
│   └── ModelLast.py         # Arsitektur MiniViT (Vision Transformer kompak)
├── Utils/
│   ├── getData.py           # Penelusuran folder 5-fold & Dataset/DataLoader PyTorch
│   ├── Preprocessing.py     # Pipeline resize, augmentasi on-the-fly & normalisasi
│   └── metrics.py           # Confusion matrix, acc, precision, recall, F1, ROC-AUC, & plot
├── results/
│   ├── SimpleCNN/           # learning_curve_fold1.png, confusion_matrix_*.png, test_metrics_*.csv
│   ├── ModifiedCNN/         # learning_curve_fold1.png, confusion_matrix_*.png, test_metrics_*.csv
│   └── MiniViT/             # learning_curve_fold1.png, confusion_matrix_*.png, test_metrics_*.csv
├── Train.py                 # Pelatihan 3 model lintas 5 fold & penyimpanan checkpoint terbaik
├── test.py                  # Evaluasi pada Original Test 5-fold & ekspor grafik/CSV
├── requirements.txt         # Daftar dependensi Python
└── README.md                # Dokumentasi proyek
```
Model terlatih (`checkpoints/*.pth`) dan dataset (`Dataset/`) **tidak** di-commit ke GitHub (diatur dalam `.gitignore`).

## 5. Cara Menjalankan
1. **Kloning repositori & instalasi dependensi:**
   ```bash
   git clone https://github.com/clarenoecho/Mpox-Skin-Lesion-Classification.git
   cd Mpox-Skin-Lesion-Classification
   pip install -r requirements.txt
   ```
2. **Siapkan folder `Dataset/`:**
   Ekstrak MSLD v2.0 sehingga memiliki struktur `Dataset/Original Images/Original Images/FOLDS/fold1..fold5` dan `Dataset/Augmented Images/Augmented Images/FOLDS_AUG/fold1..fold5`.
3. **Jalankan pelatihan dan pengujian 5-Fold:**
   ```bash
   python Train.py
   python test.py
   ```
   Seluruh grafik evaluasi (`.png`) dan rekap metrik (`.csv`) akan otomatis tersimpan di dalam folder `results/<NamaModel>/`.

## 6. Hasil
Rekap rata-rata ± simpangan baku dari pengujian 5-fold pada `Original Test` (bersumber dari `results/<Model>/test_metrics_5fold_average.csv`):

| Model | Accuracy | Precision (Macro) | Recall (Macro) | F1-Score (Macro) | ROC-AUC (OvR Macro) |
|---|---|---|---|---|---|
| **Simple CNN** | 46,18% ± 2,23% | 45,25% ± 5,64% | 38,75% ± 2,13% | 37,74% ± 2,92% | 77,40% ± 3,23% |
| **Modified CNN** | **67,95% ± 3,08%** | **66,55% ± 7,22%** | **57,02% ± 5,97%** | **57,74% ± 5,82%** | **89,43% ± 2,96%** |
| **MiniViT** | 46,33% ± 6,14% | 31,35% ± 10,69% | 31,70% ± 5,42% | 28,51% ± 5,94% | 74,08% ± 3,57% |

Visualisasi hasil tersimpan pada masing-masing direktori model di `results/<Model>/`:
- **Kurva Pembelajaran (Fold 1):** `results/SimpleCNN/learning_curve_fold1.png`, `results/ModifiedCNN/learning_curve_fold1.png`, `results/MiniViT/learning_curve_fold1.png`
- **Confusion Matrix Akumulasi 5-Fold:** `results/SimpleCNN/confusion_matrix_5fold_average.png`, `results/ModifiedCNN/confusion_matrix_5fold_average.png`, `results/MiniViT/confusion_matrix_5fold_average.png` (beserta matriks per *fold* `confusion_matrix_fold1.png` s.d. `fold5.png`).

## 7. Analysis and Discussion
1. **Distribusi data per kelas** — Dataset MSLD v2.0 memiliki ketidakseimbangan kelas yang tajam dengan **rasio 5,16 : 1** antara kelas terbanyak dan tersedikit: **Monkeypox** 284 citra original (37,62% | uji 125), **HFMD** 161 (21,32% | uji 82), **Healthy** 114 (15,10% | uji 50), **Chickenpox** 75 (9,93% | uji 42), **Cowpox** 66 (8,74% | uji 35), dan **Measles** 55 (7,28% | uji 30), dengan total **755 citra original** dan **364 sampel uji** lintas 5 *fold*. Tanpa pembobotan *loss*, dominasi Monkeypox dan HFMD mendorong batas keputusan model ke kelas mayoritas.
2. **Preprocessing & pengaruhnya** — Penggunaan skema `basic` dengan peningkatan resolusi dari $100\times100$ menjadi $128\times128$ piksel serta augmentasi *on-the-fly* (`RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomRotation`, dan `ColorJitter`) menjaga ketajaman tekstur tepi vesikel kulit dan memberikan lonjakan kinerja pada **Modified CNN** (**+21,77%** akurasi dan **+20,00%** *Macro-F1* dibanding *baseline*). Sebaliknya, pada **Simple CNN** yang tidak memiliki lapisan normalisasi maupun *dropout*, augmentasi saja tidak cukup mencegah model menghafal data latih.
3. **Apakah model mampu mengklasifikasi 6 kelas dengan baik?** — **Modified CNN** mampu mengklasifikasikan kelas mayoritas dan menengah dengan sangat baik, yaitu **HFMD** (*Prec* 70,10% / *Rec* 82,93% / *F1* 75,98%), **Healthy** (*Prec* 77,08% / *Rec* 74,00% / *F1* 75,51%), dan **Monkeypox** (*Prec* 66,04% / *Rec* 84,00% / *F1* 73,94%), serta cukup baik pada **Chickenpox** (*Prec* 74,07% / *Rec* 47,62% / *F1* 57,97%). Namun, model masih kesulitan pada dua kelas minoritas terkecil, yakni **Measles** (*F1* 37,50%) dan **Cowpox** (*F1* 32,00%). Sementara itu, **MiniViT** mengalami *majority class collapse* (menebak 211 dari 364 sampel uji sebagai Monkeypox) dengan *F1-score* **0,00%** pada Chickenpox.
4. **Kelas yang paling sering salah** — Pada **Modified CNN**, tingkat kesalahan tertinggi ($\text{Error Rate} = 1 - \text{Recall}$) terjadi pada **Cowpox** sebesar **77,14%** (27/35 salah; tertukar terbanyak ke *Monkeypox* 18x dan *HFMD* 7x), diikuti **Measles** sebesar **70,00%** (21/30 salah; tertukar ke *HFMD* 9x dan *Monkeypox* 8x), dan **Chickenpox** sebesar **52,38%** (22/42 salah; tertukar ke *Monkeypox* 17x). Pada **Simple CNN** dan **MiniViT**, kesalahan tertinggi terjadi pada **Chickenpox** (masing-masing **90,48%** dan **100,00%**). Secara klinis, kesalahan ini terjadi karena *Cowpox*, *Chickenpox*, dan *Monkeypox* sama-sama memiliki morfologi lesi vesikel/pustula bulat, sedangkan *Measles* dan *HFMD* sama-sama berupa ruam kemerahan makulopapular.
5. **Hasil tiap fold** — Konsistensi *Macro-F1* (%) pada `Original Test` di kelima *fold* menunjukkan **Modified CNN** unggul mutlak di setiap partisi:

   | Fold | Simple CNN (Acc / F1 / AUC) | Modified CNN (Acc / Prec / Rec / F1 / AUC) | MiniViT (Acc / F1 / AUC) |
   |---|---|---|---|
   | **Fold 1** | 45,95% / 37,36% / 76,77% | **66,22% / 65,74% / 52,14% / 52,53% / 91,35%** | 37,84% / 19,90% / 76,52% |
   | **Fold 2** | 47,56% / 40,46% / 75,11% | **70,73% / 71,90% / 56,57% / 59,23% / 88,32%** | 54,88% / 34,40% / 71,42% |
   | **Fold 3** | 43,75% / 35,15% / 74,38% | **67,19% / 57,03% / 50,29% / 51,22% / 85,05%** | 46,88% / 28,14% / 70,50% |
   | **Fold 4** | 44,44% / 34,72% / 78,18% | **64,20% / 62,85% / 63,69% / 60,50% / 89,69%** | 44,44% / 26,35% / 72,95% |
   | **Fold 5** | 49,21% / 41,01% / 82,55% | **71,43% / 75,22% / 62,43% / 65,19% / 92,75%** | 47,62% / 33,77% / 79,00% |
   | **Rata-Rata** | **46,18% / 37,74% / 77,40%** | **67,95% / 66,55% / 57,02% / 57,74% / 89,43%** | **46,33% / 28,51% / 74,08%** |

6. **Rata-rata 5 fold** — **Modified CNN** meraih peringkat pertama secara mutlak pada seluruh metrik (*Accuracy* **67,95%**, *Precision* **66,55%**, *Recall* **57,02%**, *Macro-F1* **57,74%**, dan *ROC-AUC* **89,43%**), diikuti **Simple CNN** di peringkat kedua dalam keseimbangan kelas (*Macro-F1* **37,74%**), dan **MiniViT** di peringkat ketiga (*Macro-F1* **28,51%**).
7. **Kelebihan & keterbatasan** — **Simple CNN** paling ringan (50.876 parameter) sehingga komputasinya paling cepat, tetapi lapisan `Flatten` (4.050 fitur) tanpa regularisasi memicu *overfitting* berat (*validation loss* melonjak dari 1,6 hingga $>3,41$). **Modified CNN** (288.230 parameter) mencapai titik keseimbangan terbaik dengan konvergensi sehat (*validation loss* turun stabil ke $\sim 1,20$ dalam 25 *epoch*) karena *inductive bias* konvolusi lokal bekerja efektif pada data berukuran kecil. Sebaliknya, **MiniViT** (1.944.006 parameter) mengalami *overfitting* setelah *epoch* ke-8 akibat ketiadaan *inductive bias* spasial lokal saat dilatih dari nol pada 755 citra *web-scraped* yang tidak seimbang.

## 8. Kesimpulan
Penelitian ini membuktikan bahwa **Modified CNN** merupakan arsitektur terbaik untuk klasifikasi enam kelas lesi kulit pada dataset MSLD v2.0 ketika dilatih dari nol dengan skema *5-fold cross validation*, mencapai rata-rata *accuracy* **67,95%**, *precision* makro **66,55%**, *recall* makro **57,02%**, *F1-score* makro **57,74%**, dan *ROC-AUC* **89,43%**, atau meningkat sebesar **+21,77%** pada akurasi dan **+20,00%** pada *Macro-F1* dibandingkan *baseline* **Simple CNN**. Peningkatan kinerja tersebut didorong oleh kombinasi resolusi masukan $128\times128$ piksel, tiga blok konvolusi ganda $3\times3$, *Batch Normalization*, aktivasi *LeakyReLU*, *Global Average Pooling*, dan *Dropout(0,5)* yang efektif menekan *overfitting*, sementara **MiniViT** kurang sesuai dilatih dari nol pada dataset medis berjumlah kecil karena mengalami bias ke kelas mayoritas. Untuk pengembangan selanjutnya, disarankan penerapan *transfer learning* dari bobot pra-latih (apabila diizinkan ketentuan tugas), penggunaan *class-weighted cross-entropy loss* atau *focal loss* untuk meningkatkan sensitivitas pada kelas minoritas (*Cowpox* dan *Measles*), eksplorasi *preprocessing* koreksi pencahayaan (CLAHE dan *gray-world*), metode *ensemble*, serta validasi klinis pada dataset eksternal dengan keragaman warna kulit yang lebih luas.

## 9. Pembagian Tugas
Program Studi Sains Data, Fakultas Informatika — Telkom University Surabaya:

| Anggota | NIM | Tanggung Jawab | File Terkait |
|---|---|---|---|
| **Fitri Nabila Khalisa Airlangga** | `108102500010` | Implementasi arsitektur `MiniViT`, pengujian dan validasi *training* & *testing*, penyusunan laporan paper (IEEE), serta dokumentasi `README.md` | `Models/ModelLast.py`, `Train.py`, `test.py`, Laporan Paper, `README.md` |
| **Mochamad Ariq Shabri Sarono** | `108102500015` | Implementasi arsitektur *baseline* `SimpleCNN` serta eksekusi penuh *training* dan *testing* 5-fold untuk seluruh model pada GPU RTX | `Models/CNN.py`, `Train.py`, `test.py`, `results/` |
| **Tasya Clara Tarigan** | `108102500039` | Pengumpulan dan penyiapan dataset MSLD v2.0, implementasi arsitektur `ModifiedCNN`, pengelolaan repositori GitHub, serta pembuatan materi presentasi (PPT) | `Models/ModifiedCNN.py`, `Utils/getData.py`, `Utils/Preprocessing.py`, GitHub Repo, Slide PPT |

## 10. Referensi
[1] S. N. Ali *et al.*, “A web-based mpox skin lesion detection system using state-of-the-art deep learning models considering racial diversity,” *arXiv:2306.14169*, 2023.  
[2] A. Dosovitskiy *et al.*, “An image is worth 16x16 words: Transformers for image recognition at scale,” in *Proc. ICLR*, 2021.  
[3] Fakultas Informatika, Telkom University, “Deep Learning Menggunakan Python,” Materi Kuliah Pertemuan 9–10, 2026.  
[4] J. Paul, “Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0),” Kaggle. [Online]. Available: https://www.kaggle.com/datasets/joydippaul/mpox-skin-lesion-dataset-version-20-msld-v20
