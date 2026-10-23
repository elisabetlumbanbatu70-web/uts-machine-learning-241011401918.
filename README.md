# UTS-Machine-Learning-241011401918

Repositori ini berisi pengerjaan UTS Machine Learning yang mencakup implementasi manual algoritma **Linear Regression** dan **Logistic Regression** menggunakan NumPy, evaluasi model, eksperimen threshold, serta perbandingannya dengan scikit-learn.

## Identitas Mahasiswa

| | |
|---|---|
| **Nama** | Elisabet Lumban Batu |
| **NIM** | 241011401918 |
| **Kelas** | 05 TPLM001 |
| **Mata Kuliah** | Machine Learning |


## Ringkasan Hasil

### Bagian A – Linear Regression (Prediksi Penjualan dari Biaya Iklan)

**Tujuan:** memodelkan pengaruh biaya iklan (juta Rp) terhadap penjualan bulanan (juta Rp) pada `Ads_Data.csv`.

**A1 – Load & Visualisasi.** Dataset berisi 40 baris dan 2 kolom. `X` (`BiayaIklan`) berbentuk 2 dimensi `(40, 1)` dan `y` (`Penjualan`) berbentuk `(40,)`. Scatter plot menunjukkan tren positif: semakin besar biaya iklan, semakin tinggi penjualan.

**A2 – Split Data.** Pembagian 80:20 (`test_size=0.2`), menghasilkan 32 data training dan 8 data testing.

**A3 – Normal Equation.** Rumus model: **ŷ = b + m·X**

| Parameter | Nilai |
|---|---|
| Intercept (b) | 29.3242 |
| Koefisien (m) | 9.0051 |

Persamaan: `ŷ = 29.3242 + 9.0051 · X`

Interpretasi: setiap kenaikan biaya iklan 1 juta Rp, penjualan diperkirakan naik sekitar 9 juta Rp. Intercept adalah perkiraan penjualan dasar saat biaya iklan nol.

**A4 – Evaluasi Model (implementasi NumPy).** Rumus: MAE = (1/n)·Σ|y − ŷ|, MSE = (1/n)·Σ(y − ŷ)², RMSE = √MSE, R² = 1 − SSres/SStot.

| Metrik | Training | Testing |
|---|---|---|
| MAE | 4.0424 | 6.3949 |
| MSE | 31.9135 | 54.5504 |
| RMSE | 5.6492 | 7.3858 |
| R² | 0.9556 | 0.8152 |

Model tergolong baik: R² testing sekitar 0,82 berarti ±82% variasi penjualan dapat dijelaskan oleh biaya iklan. Selisih R² training dan testing tidak terlalu jauh, sehingga tidak ada overfitting yang parah.

**A5 – Gradient Descent & Learning Rate** (5000 iterasi, dibandingkan dengan Normal Equation: b = 26.3450, m = 9.3631).

| Learning Rate | Intercept | Koefisien | Hasil |
|---|---|---|---|
| 0.001 | 23.8548 | 9.7426 | Terlalu lambat, belum konvergen penuh |
| 0.02 | 26.3450 | 9.3631 | Pas, sama dengan Normal Equation |
| 0.1 | ≈ −1.4×10¹⁶⁶ | ≈ −9.5×10¹⁶⁵ | Divergen (overshoot), muncul `RuntimeWarning: overflow` |

Warning *overflow* berarti nilai loss dan parameter membesar secara eksponensial hingga melampaui batas angka yang bisa disimpan komputer.

**A6 – Visualisasi & Prediksi.** Prediksi penjualan (model Normal Equation):

| Biaya Iklan | Prediksi Penjualan |
|---|---|
| 4.5 juta | 68.48 |
| 7 juta | 91.89 |
| 12 juta | 138.70 (ekstrapolasi) |

Prediksi untuk 12 juta berada di luar rentang data sehingga kurang dapat dipercaya. Dalam praktik, hubungan iklan dan penjualan cenderung mengalami *diminishing returns* atau titik jenuh, bukan garis lurus tanpa batas.

---

### Bagian B – Logistic Regression (Klasifikasi Email Spam)

**Tujuan:** memprediksi apakah sebuah email spam (1) atau bukan spam (0) berdasarkan `JumlahLink` dan `KataPromo` pada `Email_Data.csv`.

**B1 – Load & Visualisasi.** Dataset berisi 60 email: **35 Bukan Spam** dan **25 Spam**. Scatter plot menunjukkan pola yang cukup terpisah: email dengan link dan kata promo sedikit cenderung bukan spam, sedangkan yang banyak cenderung spam.

**B2 – Training (Gradient Descent).** Split 75:25, `learning_rate=0.1`, `n_iterations=10000`.

| Parameter | Nilai |
|---|---|
| Intercept (b) | −9.1327 |
| Koefisien m₁ (JumlahLink) | 0.8080 |
| Koefisien m₂ (KataPromo) | 0.7900 |

Persamaan: `P(Spam) = σ(−9.1327 + 0.8080 · JumlahLink + 0.7900 · KataPromo)`

Kurva log loss turun dan mendatar di akhir iterasi, artinya training konvergen dan stabil.

**B3 – Evaluasi (data testing, threshold 0.5).**

| Metrik | Nilai |
|---|---|
| Accuracy | 0.9333 |
| Precision | 0.8571 |
| Recall | 1.0000 |
| F1-Score | 0.9231 |

Confusion matrix (baris = aktual, kolom = prediksi):

|  | Pred. Bukan Spam | Pred. Spam |
|---|---|---|
| **Aktual Bukan Spam** | TN = 8 | FP = 1 |
| **Aktual Spam** | FN = 0 | TP = 6 |

- **False Positive (FP):** email asli salah ditandai spam.
- **False Negative (FN):** email spam lolos ke inbox.
- FP dinilai **lebih merugikan** karena email penting (tawaran kerja, tagihan, kode verifikasi) bisa tidak terbaca.

**B4 – Decision Boundary.** Garis keputusan (P = 0,5): `x₂ = −(b + m₁·x₁) / m₂`. Visualisasi menggunakan `contourf` probabilitas, garis batas, dan penanda khusus untuk data yang salah diklasifikasikan.

**B5 – Eksperimen Threshold.**

| Threshold | TN | FP | FN | TP | Precision | Recall |
|---|---|---|---|---|---|---|
| 0.3 | 8 | 1 | 0 | 6 | 0.8571 | 1.0 |
| 0.5 | 8 | 1 | 0 | 6 | 0.8571 | 1.0 |
| 0.7 | 9 | 0 | 0 | 6 | 1.0000 | 1.0 |

Menaikkan threshold membuat model lebih selektif: precision cenderung naik, recall cenderung turun. Threshold **0.7** dipilih karena menekan FP menjadi 0 pada data uji.

**B6 – Prediksi Email Baru (threshold 0.7).**

| Email | JumlahLink | KataPromo | P(Spam) | Prediksi |
|---|---|---|---|---|
| E1 | 1 | 2 | 0.0012 | Bukan Spam |
| E2 | 5 | 6 | 0.4128 | Bukan Spam |
| E3 | 9 | 10 | 0.9976 | Spam |
| E4 | 2 | 11 | 0.7637 | Spam |

---

### Bagian C – Pemahaman Konsep (ringkas)

1. **Logistic Regression = klasifikasi.** Outputnya probabilitas kelas, bukan nilai kontinu. Fungsi sigmoid memetakan hasil linear ke rentang 0–1, lalu threshold menentukan kelasnya.
2. **Normal Equation vs Gradient Descent.** Normal Equation menghitung solusi langsung secara analitik, sedangkan Gradient Descent memperbarui parameter secara iteratif. Gradient Descent lebih cocok untuk data besar atau fitur sangat banyak.
3. **Evaluasi pada data testing.** Data testing belum pernah dilihat model sehingga mengukur kemampuan generalisasi. *Overfitting* adalah kondisi model bagus di data training tetapi buruk di data baru.
4. **MSE vs log loss.** Linear Regression menghasilkan angka kontinu sehingga cocok dengan MSE. Logistic Regression menghasilkan probabilitas biner sehingga memakai log loss, yang memberi penalti besar pada prediksi yakin tetapi salah.
5. **Mengapa hasil bisa berbeda antar mahasiswa.** Perbedaan `random_state` pada pembagian data, serta pengaturan learning rate, jumlah iterasi, dan inisialisasi pada Gradient Descent.

---

### Bonus – Perbandingan dengan scikit-learn

`LinearRegression` scikit-learn menghasilkan intercept **26.3450** dan koefisien **9.3631**, identik dengan hasil Gradient Descent pada learning rate 0.02. Hal ini menunjukkan implementasi manual sudah benar dan konvergen ke solusi yang sama.

---

## Kesimpulan

- **Linear Regression:** biaya iklan berpengaruh positif terhadap penjualan dengan R² testing ≈ 0,82. Learning rate sangat menentukan keberhasilan Gradient Descent (0.001 terlalu lambat, 0.02 pas, 0.1 divergen).
- **Logistic Regression:** akurasi testing 93,33% pada threshold 0.5, dan precision 100% pada threshold 0.7 sehingga lebih aman untuk filter spam.

# uts-machine-learning-241011401918.
