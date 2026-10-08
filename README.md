# 🐧 Clustering Penguin Palmer Archipelago

Aplikasi Streamlit untuk Tugas Akhir Data Mining: mengelompokkan penguin (Adelie, Chinstrap, Gentoo)
dengan **Gaussian Mixture Model** tanpa memakai label spesies saat training.

- **Skenario A**: 4 ukuran tubuh (panjang & kedalaman paruh, panjang sirip, massa tubuh)
- **Skenario C**: skenario A + isotop nitrogen (Delta 15 N) dan karbon (Delta 13 C)
- Hasil: cluster, kemiripan dengan spesies, dan posisi titik pada peta PCA.

## Dataset
[Palmer Archipelago (Antarctica) Penguin Data](https://www.kaggle.com/parulpandey/palmer-archipelago-antarctica-penguin-data)
(Kaggle, Parul Pandey). Data asli: Dr. Kristen Gorman & Palmer Station LTER, 2007-2009, lisensi CC-0.
File ada di folder `data/`.

## Menjalankan di komputer
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Struktur
```
app.py              # aplikasi Streamlit
requirements.txt    # dependensi
data/               # penguins_size.csv, penguins_lter.csv
```
