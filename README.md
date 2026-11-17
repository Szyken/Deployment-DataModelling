# 🫁 Deteksi COVID-19 dari Citra Chest X-Ray (XGBoost & CNN)

Proyek ini merupakan implementasi pemodelan **Machine Learning** dan **Deep Learning** untuk mengklasifikasikan citra radiografi dada (*Chest X-Ray*) ke dalam dua kategori: **Normal** dan **COVID-19**. Model terbaik yang telah dilatih diintegrasikan ke dalam aplikasi web interaktif berbasis **Streamlit** untuk mendukung skrining awal secara *real-time*.

<p align="left">
  <a href="https://deployment-datamodelling-htwzjs36gyaufqdw4tkhej.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀_Live_Demo-Streamlit_App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit App" />
  </a>
</p>

---

### 🖥️ Preview Aplikasi

<!-- Ganti URL gambar di bawah dengan path screenshot di repo kamu atau drag-and-drop gambar ke editor GitHub -->
<p align="center">
  <img width="1917" height="901" alt="Screenshot 2026-11-17 145931" src="https://github.com/user-attachments/assets/5e532687-e756-472a-b4d6-85efd05df701" />
</p>

---

### 📖 Cara Menggunakan Aplikasi (How to Use)

Aplikasi web ini dirancang agar mudah digunakan oleh pengguna akhir tanpa perlu instalasi manual:
1. **Buka Aplikasi Web:** Kunjungi tautan [**Live Demo Streamlit**](https://deployment-datamodelling-htwzjs36gyaufqdw4tkhej.streamlit.app/).
2. **Unggah Citra X-Ray:** Klik tombol **Upload** pada bagian *"Upload Gambar X-Ray"* (atau *drag-and-drop* file) untuk memasukkan gambar rontgen dada berformat `JPG` atau `PNG` (maksimal 200MB).
3. **Lihat Hasil Prediksi:** Aplikasi akan otomatis menampilkan pratinjau gambar yang diunggah beserta hasil klasifikasi secara *real-time* di bagian bawah gambar (**NORMAL** dengan indikator hijau atau **COVID** dengan indikator merah).

---

### 🧠 Metodologi & Arsitektur Pemodelan

1. **Dataset & Penyeimbangan Data (Data Preparation):**
   * Menggunakan *COVID-19 Radiography Database* dari Kaggle.
   * Mengatasi ketimpangan kelas (*class imbalance*) menggunakan teknik **Random Undersampling** sehingga distribusi kelas menjadi seimbang sempurna (1:1) antara kelas Normal dan COVID-19, dilanjutkan dengan pembagian *Train-Test Split* (80:20).
   * Menerapkan **Augmentasi RGB** (rotasi, *horizontal flip*, *random brightness*, dan *zoom*) untuk memperkaya variasi spasial data latih.
2. **Eksperimen Model:**
   * **Baseline Model (XGBoost):** Mengubah matriks citra berukuran `100 x 100 x 3` menjadi vektor 1 dimensi (30.000 fitur) melalui proses *flattening* dan menormalisasi piksel ke rentang `[0, 1]`.
   * **Advanced Model (Convolutional Neural Network / CNN):** Mengekstraksi fitur spasial secara langsung menggunakan arsitektur sekuensial yang terdiri dari *Convolutional Layer*, *MaxPooling*, *Batch Normalization* (untuk mereduksi *brightness bias*), serta *Dense Layer* dengan fungsi aktivasi *ReLU* dan *Sigmoid*.
3. **Model Persistence & Deployment:**
   * Model akhir disimpan (`.pkl`) dan di-*deploy* ke dalam antarmuka web menggunakan **Streamlit**.

---

### 🛠️ Tech Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white" alt="Streamlit" />
</p>
* **Carla Ophelia Tioho** — *Eksplorasi Data & Laporan*
* **Hanif Afan Harits** — *Data Preprocessing & Laporan*
* **Ahmad Fikram Salman** — *Data Preprocessing & Laporan*
