"""Aplikasi Streamlit: Clustering Penguin Palmer Archipelago (GMM)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Clustering Penguin Palmer", page_icon="🐧", layout="wide")

DATA = Path(__file__).parent / "data"
RS = 42
NUM = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
ISOTOP = ["delta_15n", "delta_13c"]
SKENARIO = {"A: morfologi saja": "A", "C: morfologi + isotop": "C"}


@st.cache_data
def muat_data():
    size = pd.read_csv(DATA / "penguins_size.csv")
    lter = pd.read_csv(DATA / "penguins_lter.csv")
    df = size.copy()
    df["sex"] = df["sex"].replace(".", np.nan)
    assert (lter["Species"].str.split().str[0].values == df["species"].values).all()
    df["delta_15n"] = lter["Delta 15 N (o/oo)"].values
    df["delta_13c"] = lter["Delta 13 C (o/oo)"].values
    df = df.dropna(subset=NUM).reset_index(drop=True)          # buang 2 baris tanpa ukuran
    dfC = df.dropna(subset=ISOTOP).reset_index(drop=True)       # skenario C: buang isotop kosong
    return df, dfC


@st.cache_resource
def latih_model():
    dfA, dfC = muat_data()
    konfigurasi = {"A": (dfA, NUM), "C": (dfC, NUM + ISOTOP)}
    model = {}
    for kode, (dfx, fitur) in konfigurasi.items():
        sc = StandardScaler().fit(dfx[fitur])
        Xs = sc.transform(dfx[fitur])
        gm = GaussianMixture(n_components=3, random_state=RS).fit(Xs)
        lab = gm.predict(Xs)
        peta = pd.crosstab(lab, dfx["species"]).idxmax(axis=1).to_dict()   # cluster -> spesies mayoritas
        pca = PCA(n_components=2, random_state=RS).fit(Xs)
        model[kode] = dict(
            fitur=fitur, scaler=sc, gmm=gm, peta=peta, pca=pca, Xp=pca.transform(Xs),
            spesies=dfx["species"].values, n=len(dfx),
            silhouette=silhouette_score(Xs, lab),
            ari=adjusted_rand_score(dfx["species"], lab),
        )
    return model


dfA, dfC = muat_data()
model = latih_model()

# ---------- nilai awal slider & contoh ----------
semua = dfC[NUM + ISOTOP]
median_spesies = dfC.groupby("species")[NUM + ISOTOP].median().round(2)
KOLOM = {
    "culmen_length_mm": ("Panjang paruh (mm)", 0.1),
    "culmen_depth_mm": ("Kedalaman paruh (mm)", 0.1),
    "flipper_length_mm": ("Panjang sirip (mm)", 1.0),
    "body_mass_g": ("Massa tubuh (g)", 25.0),
    "delta_15n": ("Delta 15 N (isotop nitrogen)", 0.01),
    "delta_13c": ("Delta 13 C (isotop karbon)", 0.01),
}
for k in KOLOM:
    st.session_state.setdefault(k, float(round(semua[k].median(), 2)))


def pakai_contoh(spesies):
    for k in KOLOM:
        st.session_state[k] = float(median_spesies.loc[spesies, k])


# ---------- tampilan ----------
st.title("🐧 Clustering Penguin Palmer Archipelago")
st.caption("Model: Gaussian Mixture (3 cluster), unsupervised. Nama spesies hanya interpretasi cluster "
           "(spesies mayoritas di tiap cluster).")

tab_prediksi, tab_model = st.tabs(["Prediksi", "Tentang model"])

with st.sidebar:
    st.header("Input penguin")
    pilihan = st.radio("Skenario fitur", list(SKENARIO), index=1)
    kode = SKENARIO[pilihan]
    st.markdown("**Isi cepat dengan contoh (median spesies):**")
    c1, c2, c3 = st.columns(3)
    for kol, sp in zip((c1, c2, c3), ("Adelie", "Chinstrap", "Gentoo")):
        kol.button(sp, on_click=pakai_contoh, args=(sp,))

    nilai = {}
    for k, (label, step) in KOLOM.items():
        pakai = k in NUM or kode == "C"
        lo, hi = float(np.floor(semua[k].min())), float(np.ceil(semua[k].max()))
        nilai[k] = st.slider(label + ("" if pakai else " (tidak dipakai di skenario A)"),
                             lo, hi, step=step, key=k, disabled=not pakai)

m = model[kode]
x = m["scaler"].transform(pd.DataFrame([[nilai[k] for k in m["fitur"]]], columns=m["fitur"]))
prob = m["gmm"].predict_proba(x)[0]
c = int(prob.argmax())
probabilitas = pd.Series({m["peta"][i]: float(prob[i]) for i in range(3)}).sort_values(ascending=False)
titik = m["pca"].transform(x)[0]

with tab_prediksi:
    kiri, kanan = st.columns([1, 1.2])
    with kiri:
        st.metric("Cluster", f"Cluster {c}")
        st.metric("Paling mirip dengan", m["peta"][c], f"keyakinan {prob[c] * 100:.1f}%", delta_color="off")
        st.subheader("Kemiripan dengan tiap spesies")
        for sp, p in probabilitas.items():
            st.write(f"{sp}: **{p * 100:.1f}%**")
            st.progress(min(max(p, 0.0), 1.0))
    with kanan:
        fig, ax = plt.subplots(figsize=(6, 4.6))
        sns.scatterplot(x=m["Xp"][:, 0], y=m["Xp"][:, 1], hue=m["spesies"], alpha=0.6, ax=ax)
        ax.scatter(*titik, marker="*", s=420, c="red", edgecolor="black", label="Input kamu", zorder=5)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_title("Posisi pada peta PCA"); ax.legend(fontsize=8)
        st.pyplot(fig)
        plt.close(fig)

with tab_model:
    st.markdown(
        "**Dataset:** Palmer Archipelago (Antarctica) Penguin Data (Kaggle, Parul Pandey; data asli Dr. Kristen "
        "Gorman & Palmer Station LTER, 2007-2009, lisensi CC-0).\n\n"
        "**Skenario fitur:** A = 4 ukuran tubuh; C = A + isotop nitrogen & karbon."
    )
    st.dataframe(pd.DataFrame({
        "Skenario": ["A: morfologi saja", "C: morfologi + isotop"],
        "Jumlah data": [model["A"]["n"], model["C"]["n"]],
        "Silhouette": [round(model["A"]["silhouette"], 3), round(model["C"]["silhouette"], 3)],
        "ARI vs spesies asli": [round(model["A"]["ari"], 3), round(model["C"]["ari"], 3)],
    }), hide_index=True)
    st.markdown("**Peta cluster → spesies:**")
    st.json({f"Skenario {k}": {f"Cluster {i}": s for i, s in model[k]["peta"].items()} for k in model})
