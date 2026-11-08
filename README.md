# Heart Disease Predictor

Aplikasi prediksi risiko penyakit gagal jantung berbasis Machine Learning menggunakan Random Forest Classifier dengan antarmuka Streamlit.

## Bahasa Pemrograman

- Python 3.10+


## Framework dan Library

Machine Learning:
- Scikit-learn
- Pandas
- NumPy
- Joblib

Web Framework:
- Streamlit

Visualisasi:
- Plotly
- Matplotlib

Lainnya:
- PIL (Pillow)
- base64


## Fungsi dan Fitur

1. Input Data Pasien
   - 11 parameter medis (Usia, Tekanan Darah, Kolesterol, dll)

2. Prediksi Real-time
   - Hasil prediksi langsung dengan probabilitas (%)

3. Visualisasi
   - Bar chart probabilitas
   - Gauge chart risiko

4. Feature Importance
   - Grafik 10 fitur paling berpengaruh

5. Saran Kesehatan
   - Rekomendasi berdasarkan hasil prediksi

6. Statistik Dataset
   - Informasi jumlah sampel, fitur, dan distribusi


## Kelebihan

1. Antarmuka user-friendly dan responsif
2. Akurasi model sekitar 88%
3. Visualisasi interaktif dengan Plotly
4. Caching untuk performa lebih cepat
5. Saran kesehatan yang bermanfaat
6. Open source dan mudah dikembangkan


## Kekurangan dan Bug

1. Dataset terbatas (918 sampel) - Warning
2. Tidak ada validasi input - Warning
3. Tidak ada autentikasi - Warning
4. Hanya binary classification - Warning
5. Model statis (tidak belajar dari data baru) - Warning
6. Tidak ada dark mode - Warning
7. Tidak ada fitur download hasil - Warning
8. Error handling minim - Bug


## Dataset

Link Dataset:
- Kaggle: https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction

Penjelasan Dataset:

Dataset ini merupakan gabungan dari 5 sumber data:
- Cleveland: 303 sampel
- Hungarian: 294 sampel
- Switzerland: 123 sampel
- Long Beach VA: 200 sampel
- Stalog (Heart): 270 sampel

Total sampel setelah menghapus duplikat: 918 sampel

Fitur dataset (11 fitur + 1 target):

1. Age - Usia pasien (tahun) - Numerik
2. Sex - Jenis kelamin (M/F) - Kategorikal
3. ChestPainType - Tipe nyeri dada (ATA, NAP, ASY, TA) - Kategorikal
4. RestingBP - Tekanan darah istirahat (mmHg) - Numerik
5. Cholesterol - Kolesterol (mg/dl) - Numerik
6. FastingBS - Gula darah puasa > 120 (0/1) - Kategorikal
7. RestingECG - Hasil EKG (Normal, ST, LVH) - Kategorikal
8. MaxHR - Detak jantung maksimum - Numerik
9. ExerciseAngina - Angina saat olahraga (Y/N) - Kategorikal
10. Oldpeak - Depresi ST - Numerik
11. ST_Slope - Kemiringan ST (Up, Flat, Down) - Kategorikal
12. HeartDisease - Target variable (0 = Tidak sakit, 1 = Sakit) - Binary

Distribusi target:
- Tidak sakit (0): 419 (45.6%)
- Sakit (1): 499 (54.4%)

