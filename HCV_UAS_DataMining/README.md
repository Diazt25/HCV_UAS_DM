# 🩸 HCV Liver Disease Detection System

Proyek Data Mining — Deteksi Dini Penyakit Hati Berbasis Biomarker Darah

## Dataset
- **Nama:** HCV (Hepatitis C Virus) Dataset
- **Sumber:** UCI Machine Learning Repository
- **Records:** 615 | **Fitur:** 12 (Age, Sex, 10 biomarker darah)
- **Target:** Binary — Donor Sehat (0) vs Indikasi Penyakit Hati (1)

## Metode
1. **Random Forest Classification** — prediksi status pasien
2. **K-Means Clustering** — segmentasi profil biomarker pasien

## Struktur Folder
```
UAS_DataMining_NamaKelompok/
├── dataset/
│   └── hcvdat0.csv
├── notebook/
│   └── hcv_analysis.ipynb
├── model/
│   ├── model_hcv.pkl
│   ├── scaler_hcv.pkl
│   ├── imputer_hcv.pkl
│   ├── kmeans_hcv.pkl
│   └── pca_hcv.pkl
├── app/
│   ├── app.py
│   └── cluster_data.csv
├── requirements.txt
└── README.md
```

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan notebook (generate model)
```bash
jupyter notebook hcv_analysis.ipynb
```
Jalankan semua cell. File `.pkl` akan ter-generate otomatis.

### 3. Jalankan Streamlit
```bash
streamlit run app.py
```

## Hasil Model
| Metode | Metrik | Nilai |
|--------|--------|-------|
| Random Forest | ROC-AUC | 0.9963 |
| Random Forest | F1-Score Macro | 0.9376 |
| K-Means (k=3) | Silhouette Score | 0.2123 |
