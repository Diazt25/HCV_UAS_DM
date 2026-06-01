import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="HCV Liver Disease Detector",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 50%, #1a2332 100%);
    border: 1px solid #30363d;
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(220,38,38,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.main-header h1 {
    margin: 0;
    font-size: 2rem;
    font-weight: 700;
    color: #f0f6fc !important;
    letter-spacing: -0.5px;
}
.main-header .subtitle {
    margin: 0.5rem 0 0;
    color: #8b949e !important;
    font-size: 0.9rem;
}
.main-header .badge {
    display: inline-block;
    background: rgba(220,38,38,0.2);
    border: 1px solid rgba(220,38,38,0.4);
    color: #f87171;
    padding: 0.15rem 0.6rem;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-right: 0.5rem;
    font-family: 'DM Mono', monospace;
}

.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.metric-card .label {
    font-size: 0.72rem;
    color: #8b949e;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: .07em;
    font-family: 'DM Mono', monospace;
}
.metric-card .value {
    font-size: 1.9rem;
    font-weight: 700;
    color: #f0f6fc;
    line-height: 1.2;
    margin-top: 0.2rem;
}
.metric-card .sub {
    font-size: 0.78rem;
    color: #8b949e;
    margin-top: 0.2rem;
}

.result-healthy {
    background: linear-gradient(135deg, #064e3b, #065f46);
    border: 1px solid #10b981;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: white;
}
.result-sick {
    background: linear-gradient(135deg, #7f1d1d, #991b1b);
    border: 1px solid #ef4444;
    border-radius: 14px;
    padding: 1.8rem;
    text-align: center;
    color: white;
}
.result-icon { font-size: 3rem; margin-bottom: 0.5rem; }
.result-label { font-size: 1.4rem; font-weight: 700; margin-bottom: 0.3rem; }
.result-desc { font-size: 0.88rem; opacity: 0.85; }

.prob-row { margin: 0.5rem 0; }
.prob-label {
    font-size: 0.83rem;
    font-weight: 600;
    color: #c9d1d9;
    font-family: 'DM Mono', monospace;
    margin-bottom: 4px;
}
.prob-bar-bg {
    background: #21262d;
    border-radius: 20px;
    height: 12px;
    overflow: hidden;
}
.prob-bar-fill { height: 100%; border-radius: 20px; }

.info-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #c9d1d9;
}
.info-box b { color: #f0f6fc; }

.section-title {
    font-size: 1.2rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #21262d;
}

.cluster-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}
.cluster-card .cluster-name {
    font-size: 1rem;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 0.5rem;
}
.cluster-card .cluster-stat {
    font-size: 0.82rem;
    color: #8b949e;
    font-family: 'DM Mono', monospace;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #0d1117;
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
section[data-testid="stSidebar"] .stSlider > div > div { background: #21262d; }

div.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #dc2626, #b91c1c);
    color: white;
    border: none;
    padding: 0.75rem;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.02em;
    cursor: pointer;
    transition: opacity 0.2s;
}
div.stButton > button:hover { opacity: 0.88; }

.stTabs [data-baseweb="tab-list"] { background: #161b22; border-radius: 10px; padding: 4px; }
.stTabs [data-baseweb="tab"] { color: #8b949e; font-weight: 600; }
.stTabs [aria-selected="true"] { background: #21262d; border-radius: 8px; color: #f0f6fc !important; }
</style>
""", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model   = joblib.load("model_hcv.pkl")
    scaler  = joblib.load("scaler_hcv.pkl")
    imputer = joblib.load("imputer_hcv.pkl")
    kmeans  = joblib.load("kmeans_hcv.pkl")
    pca     = joblib.load("pca_hcv.pkl")
    return model, scaler, imputer, kmeans, pca

@st.cache_data
def load_cluster_data():
    return pd.read_csv("cluster_data.csv")

@st.cache_data
def load_raw_data():
    df = pd.read_csv("hcvdat0.csv").drop(columns=["Unnamed: 0"])
    df["target"] = df["Category"].apply(lambda x: 0 if "0" in str(x) else 1)
    return df

try:
    model, scaler, imputer, kmeans, pca = load_models()
    df_cluster = load_cluster_data()
    df_raw = load_raw_data()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False
    df_raw = load_raw_data()

FEATURES = ['Age','Sex_enc','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
BIOMARKERS = ['ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩸 Input Data Pasien")
    st.markdown("Masukkan nilai biomarker darah pasien.")
    st.markdown("---")

    age = st.slider("Usia (tahun)", 19, 77, 45)
    sex = st.radio("Jenis Kelamin", ["Laki-laki (m)", "Perempuan (f)"], horizontal=True)
    sex_enc = 1 if sex.startswith("Laki") else 0

    st.markdown("**🔬 Biomarker Hati**")
    alb  = st.slider("ALB — Albumin (g/L)",       14.9, 82.2, 41.6, 0.1)
    alp  = st.slider("ALP — Alkaline Phosphatase", 11.3, 416.6, 68.3, 0.1)
    alt  = st.slider("ALT — Alanine Aminotransf.", 0.9, 325.3, 28.5, 0.1)
    ast  = st.slider("AST — Aspartate Aminotransf.", 10.6, 324.0, 34.8, 0.1)
    bil  = st.slider("BIL — Bilirubin (µmol/L)",   0.8, 254.0, 11.4, 0.1)
    che  = st.slider("CHE — Cholinesterase",        1.4, 16.4, 8.2, 0.1)
    chol = st.slider("CHOL — Cholesterol (mmol/L)", 1.4, 9.7, 5.4, 0.1)
    crea = st.slider("CREA — Creatinine (µmol/L)",  8.0, 1079.1, 81.3, 1.0)
    ggt  = st.slider("GGT — Gamma-Glutamyl Trans.", 4.5, 650.9, 39.5, 0.1)
    prot = st.slider("PROT — Total Protein (g/L)",  44.8, 90.0, 72.0, 0.1)

    st.markdown("---")
    predict_btn = st.button("🔍 Analisis Sekarang")

    st.markdown("---")
    st.markdown("""
    <div class="info-box">
    <b>📖 Tentang Model</b><br>
    Algoritma: <b>Random Forest</b><br>
    Handling Imbalance: <b>SMOTE</b><br>
    Dataset: <b>HCV UCI (615 data)</b><br>
    ROC-AUC: <b>0.9963</b><br>
    F1-Score Macro: <b>0.9376</b>
    </div>
    """, unsafe_allow_html=True)

# ─── Header ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div>
        <span class="badge">Classification</span>
        <span class="badge">Clustering</span>
        <span class="badge">Random Forest</span>
        <span class="badge">K-Means</span>
    </div>
    <h1>🩸 HCV Liver Disease Detection System</h1>
    <p class="subtitle">Data Mining — Deteksi Dini Penyakit Hati Berbasis Biomarker Darah &nbsp;|&nbsp; Dataset: UCI HCV</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ File model tidak ditemukan! Jalankan notebook `hcv_analysis.ipynb` terlebih dahulu.")
    st.stop()

# ─── Navigasi Tab ─────────────────────────────────────────────────────────────
tab_home, tab_data, tab_predict, tab_cluster, tab_about = st.tabs([
    "🏠 Home", "📊 Dataset", "🔍 Prediksi", "🔵 Clustering", "📖 About"
])

# ══════════════════════════════════════════════════════════
# TAB 1 — HOME
# ══════════════════════════════════════════════════════════
with tab_home:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="label">Total Pasien</div>
            <div class="value">615</div>
            <div class="sub">Records dalam dataset</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="label">Fitur Biomarker</div>
            <div class="value">12</div>
            <div class="sub">Termasuk usia & jenis kelamin</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="label">ROC-AUC Score</div>
            <div class="value">99.6%</div>
            <div class="sub">Random Forest + SMOTE</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="label">F1-Score Macro</div>
            <div class="value">93.8%</div>
            <div class="sub">Evaluasi multi-kelas</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        st.markdown('<div class="section-title">🎯 Tentang Proyek</div>', unsafe_allow_html=True)
        st.markdown("""
        Proyek ini mengimplementasikan dua metode Data Mining untuk **deteksi dini penyakit hati**
        berbasis data biomarker darah pasien dari dataset HCV (Hepatitis C Virus) yang bersumber dari
        UCI Machine Learning Repository.

        **Dua metode yang diterapkan:**

        - **Classification (Random Forest):** Memprediksi apakah seorang pasien termasuk *Donor Sehat*
          atau memiliki *Indikasi Penyakit Hati* (Hepatitis, Fibrosis, atau Cirrhosis).

        - **Clustering (K-Means):** Mengelompokkan pasien berdasarkan pola biomarker secara unsupervised
          untuk menemukan segmen alami dalam populasi pasien.

        **Tantangan utama yang ditangani:** Class imbalance ekstrem (540 sehat vs 75 sakit) diatasi
        menggunakan teknik **SMOTE** (Synthetic Minority Over-sampling Technique).
        """)

    with col_b:
        st.markdown('<div class="section-title">📋 Distribusi Kategori</div>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        cats = df_raw['Category'].value_counts()
        colors = ['#10b981','#059669','#ef4444','#dc2626','#b91c1c']
        bars = ax.barh(cats.index, cats.values, color=colors[:len(cats)], edgecolor='#21262d', linewidth=1.5)
        for bar, val in zip(bars, cats.values):
            ax.text(val + 3, bar.get_y() + bar.get_height()/2, str(val),
                    va='center', fontsize=9, fontweight='bold', color='#c9d1d9')
        ax.set_xlabel('Jumlah', color='#8b949e')
        ax.tick_params(colors='#8b949e')
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.set_title('Distribusi Kategori Pasien', color='#f0f6fc', fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 2 — DATASET
# ══════════════════════════════════════════════════════════
with tab_data:
    st.markdown('<div class="section-title">📊 Dataset Overview</div>', unsafe_allow_html=True)

    sub1, sub2 = st.tabs(["📈 Distribusi Biomarker", "🔥 Korelasi"])

    with sub1:
        st.markdown("**Distribusi 6 biomarker utama berdasarkan status pasien**")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        fig.patch.set_facecolor('#0d1117')
        axes = axes.flatten()
        bio_sel = ['ALT','AST','GGT','BIL','ALB','CHE']

        df_bio = df_raw[bio_sel + ['target']].copy()

        for i, feat in enumerate(bio_sel):
            axes[i].set_facecolor('#161b22')
            for label, color, name in [(0,'#10b981','Donor Sehat'), (1,'#ef4444','Penyakit Hati')]:
                data = df_bio[df_bio['target']==label][feat].dropna()
                axes[i].hist(data, bins=25, alpha=0.65, color=color,
                             label=name, edgecolor='#0d1117', linewidth=0.5)
            axes[i].set_title(feat, fontweight='bold', color='#f0f6fc', fontsize=10)
            axes[i].legend(fontsize=7)
            axes[i].tick_params(colors='#8b949e', labelsize=8)
            for spine in axes[i].spines.values():
                spine.set_edgecolor('#30363d')

        plt.suptitle('Distribusi Biomarker: Sehat vs Penyakit Hati', color='#f0f6fc',
                     fontweight='bold', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    with sub2:
        df_num = df_raw[BIOMARKERS + ['Age']].dropna()
        fig, ax = plt.subplots(figsize=(10, 7))
        fig.patch.set_facecolor('#161b22')
        ax.set_facecolor('#161b22')
        corr = df_num.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, center=0,
                    mask=mask, ax=ax, linewidths=0.5,
                    annot_kws={'size': 9, 'color': 'white'},
                    cbar_kws={'shrink': 0.8})
        ax.set_title('Heatmap Korelasi Fitur', color='#f0f6fc', fontweight='bold')
        ax.tick_params(colors='#c9d1d9')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("**🗂️ Sample Dataset**")
    n_rows = st.slider("Baris ditampilkan:", 5, 30, 10)
    display_cols = ['Category','Age','Sex','ALB','ALP','ALT','AST','BIL','CHE','CHOL','CREA','GGT','PROT']
    st.dataframe(df_raw[display_cols].head(n_rows), use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 3 — PREDIKSI
# ══════════════════════════════════════════════════════════
with tab_predict:
    st.markdown('<div class="section-title">🔍 Prediksi Status Pasien</div>', unsafe_allow_html=True)
    st.markdown("Atur nilai biomarker di sidebar kiri, lalu tekan tombol **Analisis Sekarang**.")

    # Selalu hitung prediksi (bukan hanya saat tombol ditekan)
    input_arr = np.array([[age, sex_enc, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])
    input_imp = imputer.transform(input_arr)
    input_sc  = scaler.transform(input_imp)
    pred      = model.predict(input_sc)[0]
    proba     = model.predict_proba(input_sc)[0]

    col_res, col_info = st.columns([1, 1.3])

    with col_res:
        if pred == 0:
            st.markdown(f"""
            <div class="result-healthy">
                <div class="result-icon">✅</div>
                <div class="result-label">Donor Sehat</div>
                <div class="result-desc">Profil biomarker pasien menunjukkan kondisi hati yang normal.<br>
                Probabilitas sehat: <b>{proba[0]*100:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-sick">
                <div class="result-icon">⚠️</div>
                <div class="result-label">Indikasi Penyakit Hati</div>
                <div class="result-desc">Profil biomarker menunjukkan kemungkinan gangguan fungsi hati.<br>
                Probabilitas indikasi: <b>{proba[1]*100:.1f}%</b></div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Probability bars
        st.markdown("**Distribusi Probabilitas:**")
        labels = ['Donor Sehat', 'Penyakit Hati']
        colors = ['#10b981', '#ef4444']
        prob_html = ""
        for lbl, prob, color in zip(labels, proba, colors):
            pct = prob * 100
            prob_html += f"""
            <div class="prob-row">
                <div class="prob-label">{lbl} — {pct:.1f}%</div>
                <div class="prob-bar-bg">
                    <div class="prob-bar-fill" style="width:{pct}%; background:{color};"></div>
                </div>
            </div>"""
        st.markdown(prob_html, unsafe_allow_html=True)

    with col_info:
        st.markdown("**📋 Data Input Pasien:**")
        input_df = pd.DataFrame({
            "Parameter": ["Usia","Jenis Kelamin","ALB","ALP","ALT","AST","BIL","CHE","CHOL","CREA","GGT","PROT"],
            "Nilai":     [age, sex.split()[0], alb, alp, alt, ast, bil, che, chol, crea, ggt, prot],
            "Satuan":    ["tahun","-","g/L","U/L","U/L","U/L","µmol/L","kU/L","mmol/L","µmol/L","U/L","g/L"]
        })
        st.dataframe(input_df, use_container_width=True, hide_index=True)

        # Feature importance chart
        st.markdown("**🔑 Feature Importance Model:**")
        fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True)
        fig_fi, ax_fi = plt.subplots(figsize=(6, 4))
        fig_fi.patch.set_facecolor('#161b22')
        ax_fi.set_facecolor('#161b22')
        colors_fi = ['#ef4444' if v >= fi.quantile(0.75) else '#3b82f6' for v in fi.values]
        fi.plot(kind='barh', ax=ax_fi, color=colors_fi, edgecolor='#0d1117', linewidth=0.5)
        ax_fi.set_xlabel('Importance', color='#8b949e')
        ax_fi.tick_params(colors='#c9d1d9', labelsize=8)
        for spine in ax_fi.spines.values():
            spine.set_edgecolor('#30363d')
        ax_fi.set_title('Feature Importance', color='#f0f6fc', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig_fi, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 4 — CLUSTERING
# ══════════════════════════════════════════════════════════
with tab_cluster:
    st.markdown('<div class="section-title">🔵 Segmentasi Pasien — K-Means Clustering</div>', unsafe_allow_html=True)

    n_clusters = df_cluster['cluster'].nunique()
    cluster_colors_map = {0: '#3b82f6', 1: '#f59e0b', 2: '#a78bfa', 3: '#10b981'}

    col_plot, col_prof = st.columns([1.3, 1])

    with col_plot:
        st.markdown("**Visualisasi Cluster (PCA 2D)**")
        fig_cl, axes_cl = plt.subplots(1, 2, figsize=(12, 5))
        fig_cl.patch.set_facecolor('#0d1117')

        for ax_i in axes_cl:
            ax_i.set_facecolor('#161b22')
            for spine in ax_i.spines.values():
                spine.set_edgecolor('#30363d')
            ax_i.tick_params(colors='#8b949e')

        # By cluster
        for cl in range(n_clusters):
            mask = df_cluster['cluster'] == cl
            axes_cl[0].scatter(df_cluster[mask]['pca1'], df_cluster[mask]['pca2'],
                               c=cluster_colors_map[cl], label=f'Cluster {cl}',
                               alpha=0.75, s=35, edgecolors='none')
        axes_cl[0].set_title('Cluster K-Means', color='#f0f6fc', fontweight='bold')
        axes_cl[0].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)
        axes_cl[0].set_xlabel('PC 1', color='#8b949e')
        axes_cl[0].set_ylabel('PC 2', color='#8b949e')

        # By actual label
        for label, color, name in [(0,'#10b981','Donor Sehat'), (1,'#ef4444','Penyakit Hati')]:
            mask = df_cluster['target'] == label
            axes_cl[1].scatter(df_cluster[mask]['pca1'], df_cluster[mask]['pca2'],
                               c=color, label=name, alpha=0.75, s=35, edgecolors='none')
        axes_cl[1].set_title('Label Aktual', color='#f0f6fc', fontweight='bold')
        axes_cl[1].legend(facecolor='#21262d', labelcolor='#c9d1d9', fontsize=8)
        axes_cl[1].set_xlabel('PC 1', color='#8b949e')
        axes_cl[1].set_ylabel('PC 2', color='#8b949e')

        plt.tight_layout()
        st.pyplot(fig_cl, use_container_width=True)
        st.caption("PCA mereduksi 12 dimensi menjadi 2 komponen utama untuk visualisasi.")

    with col_prof:
        st.markdown("**Profil Biomarker per Cluster**")
        profile_features = ['ALT','AST','GGT','BIL','ALB','CHE']
        for cl in range(n_clusters):
            mask = df_cluster['cluster'] == cl
            cl_data = df_cluster[mask]
            n_sick = (cl_data['target'] == 1).sum()
            n_total = len(cl_data)
            sick_pct = n_sick / n_total * 100

            # Determine cluster interpretation
            mean_alt = cl_data['ALT'].mean()
            mean_alb = cl_data['ALB'].mean()
            if sick_pct > 40:
                interp = "🔴 Dominan Penyakit Hati"
            elif sick_pct > 10:
                interp = "🟡 Campuran / Perlu Perhatian"
            else:
                interp = "🟢 Dominan Donor Sehat"

            st.markdown(f"""
            <div class="cluster-card" style="border-left: 4px solid {cluster_colors_map[cl]};">
                <div class="cluster-name">Cluster {cl} — {n_total} pasien</div>
                <div class="cluster-stat">
                    Indikasi Sakit: {n_sick} ({sick_pct:.1f}%) &nbsp;|&nbsp; {interp}<br>
                    ALT (avg): {mean_alt:.1f} &nbsp;|&nbsp; ALB (avg): {mean_alb:.1f}
                </div>
            </div>""", unsafe_allow_html=True)

    # Tabel profil lengkap
    st.markdown("---")
    st.markdown("**📊 Rata-Rata Biomarker per Cluster**")
    profile_table = df_cluster.groupby('cluster')[profile_features + ['target']].agg({
        **{f: 'mean' for f in profile_features},
        'target': ['sum', 'count']
    }).round(2)
    profile_table.columns = profile_features + ['Jumlah Sakit', 'Total Pasien']
    profile_table.index.name = 'Cluster'
    st.dataframe(profile_table, use_container_width=True)

# ══════════════════════════════════════════════════════════
# TAB 5 — ABOUT
# ══════════════════════════════════════════════════════════
with tab_about:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">🧬 Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
        **HCV — Hepatitis C Virus Dataset**

        Sumber: UCI Machine Learning Repository

        Dataset ini berisi hasil pemeriksaan darah pasien dari klinik,
        mencakup biomarker fungsi hati dan metabolisme. Target klasifikasi
        mencakup 5 kategori asli:

        - `0=Blood Donor` — Donor darah sehat
        - `0s=suspect Blood Donor` — Suspect donor
        - `1=Hepatitis` — Hepatitis C akut
        - `2=Fibrosis` — Fibrosis hati
        - `3=Cirrhosis` — Sirosis hati

        Pada proyek ini, 5 kelas disederhanakan menjadi **binary classification**:
        - **Kelas 0:** Donor Sehat (0 + 0s)
        - **Kelas 1:** Indikasi Penyakit Hati (1 + 2 + 3)
        """)

        st.markdown('<div class="section-title">🔬 Biomarker</div>', unsafe_allow_html=True)
        bio_info = {
            "ALB": "Albumin — protein utama plasma darah, rendah = disfungsi hati",
            "ALP": "Alkaline Phosphatase — enzim, tinggi = obstruksi bilier",
            "ALT": "Alanine Aminotransferase — enzim hati, tinggi = kerusakan hati",
            "AST": "Aspartate Aminotransferase — enzim, tinggi = kerusakan sel hati",
            "BIL": "Bilirubin — pigmen empedu, tinggi = gangguan metabolisme",
            "CHE": "Cholinesterase — enzim, rendah = penyakit hati lanjut",
            "CHOL": "Cholesterol — lemak darah, bervariasi pada penyakit hati",
            "CREA": "Creatinine — fungsi ginjal, relevan pada komplikasi",
            "GGT": "Gamma-Glutamyl Transferase — enzim sensitif penyakit hati",
            "PROT": "Total Protein — mencerminkan fungsi sintetik hati",
        }
        for k, v in bio_info.items():
            st.markdown(f"- **{k}:** {v}")

    with col2:
        st.markdown('<div class="section-title">⚙️ Metodologi</div>', unsafe_allow_html=True)
        st.markdown("""
        **Kerangka: CRISP-DM**

        1. **Business Understanding** — Deteksi dini penyakit hati dari biomarker
        2. **Data Understanding** — EDA distribusi, korelasi, missing values
        3. **Data Preparation** — Imputasi median, encoding, SMOTE, StandardScaler
        4. **Modeling** — Random Forest (Classification) + K-Means (Clustering)
        5. **Evaluation** — F1 Macro, ROC-AUC, Silhouette Score
        6. **Deployment** — Streamlit Web Application

        ---

        **Metode 1 — Random Forest Classifier**

        Ensemble berbasis decision tree. Dipilih karena:
        - Robust terhadap outlier (biomarker sering memiliki nilai ekstrem)
        - Tidak memerlukan asumsi distribusi data
        - Menghasilkan feature importance secara langsung
        - Kombinasi dengan SMOTE efektif untuk class imbalance

        **Evaluasi:** F1-Score Macro = **0.9376** | ROC-AUC = **0.9963**

        ---

        **Metode 2 — K-Means Clustering**

        Unsupervised clustering berbasis centroid. Dipilih karena:
        - Mengungkap segmen alami pasien tanpa label
        - k=3 optimal berdasarkan Elbow Method + Silhouette Score
        - Visualisasi PCA 2D memudahkan interpretasi klinis

        **Evaluasi:** Silhouette Score = **0.2123** (struktur cluster lemah → wajar untuk data medis multidimensi)
        """)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#484f58; font-size:0.8rem; padding:0.5rem 0 1rem; font-family: DM Mono, monospace;'>
    🩸 HCV Liver Disease Detection &nbsp;|&nbsp; Data Mining &nbsp;|&nbsp;
    Random Forest + K-Means &nbsp;|&nbsp; Dataset: UCI HCV Repository
</div>
""", unsafe_allow_html=True)
