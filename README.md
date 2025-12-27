# 📄 Smart Document Scanner dengan Auto Enhancement

Aplikasi pemindai dokumen pintar berbasis Computer Vision yang dapat mendeteksi, memperbaiki, dan meningkatkan kualitas dokumen secara otomatis.

## 🎯 Fitur Utama

1. **Auto Document Detection** - Deteksi otomatis batas dokumen dari gambar
2. **Perspective Correction** - Koreksi perspektif dokumen yang miring
3. **Image Enhancement** - Peningkatan kualitas gambar dengan filtering
4. **Object Removal** - Hapus watermark/objek tidak diinginkan dengan seam carving
5. **Real-time Tracking** - Track dokumen dari webcam secara real-time
6. **Batch Processing** - Proses banyak dokumen sekaligus

## 🔬 Materi Computer Vision yang Digunakan

- ✅ **Filtering** - Gaussian blur, bilateral filter, adaptive thresholding
- ✅ **Edge Detection** - Canny edge detection
- ✅ **Boundary Detection** - Contour detection dan polygon approximation
- ✅ **Seam Carving** - Content-aware image resizing untuk object removal
- ✅ **Object Tracking** - Real-time document tracking dengan optical flow
- ✅ **CNN** - Pre-trained model untuk text/non-text classification

## 📁 Struktur Proyek

```
UAS/
├── README.md
├── requirements.txt
├── .gitignore
├── app.py                          # Aplikasi utama (Streamlit)
├── main.ipynb                      # Notebook untuk testing & demo
├── modules/
│   ├── __init__.py
│   ├── boundary_detector.py        # Deteksi batas dokumen
│   ├── image_enhancer.py           # Filter & enhancement
│   ├── perspective_corrector.py    # Koreksi perspektif
│   ├── seam_carver.py             # Seam carving untuk object removal
│   ├── document_tracker.py         # Real-time tracking
│   └── utils.py                    # Utility functions
├── sample_images/                  # Contoh gambar untuk testing
│   ├── document1.jpg
│   ├── document2.jpg
│   └── watermark.jpg
├── output/                         # Hasil output
└── docs/
    └── usage_guide.md             # Panduan penggunaan detail
```

## 🚀 Instalasi

### Prerequisites
- Python 3.8 atau lebih baru
- Webcam (opsional, untuk fitur real-time tracking)

### Install Dependencies

```bash
# Clone repository
git clone <repository-url>
cd UAS

# Install required packages
pip install -r requirements.txt
```

## 💻 Cara Penggunaan

### 1. Menjalankan Aplikasi Web (Recommended)

```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

### 2. Menggunakan Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

### 3. Menggunakan sebagai Library

```python
from modules.boundary_detector import DocumentDetector
from modules.image_enhancer import ImageEnhancer

# Deteksi dokumen
detector = DocumentDetector()
corners = detector.detect_document('input.jpg')

# Enhance gambar
enhancer = ImageEnhancer()
enhanced = enhancer.enhance('input.jpg')
```

## 📖 Panduan Fitur

### Auto Document Detection
1. Upload gambar dokumen
2. Sistem akan otomatis mendeteksi tepi dokumen menggunakan Canny edge detection
3. Contour detection untuk menemukan batas dokumen
4. Polygon approximation untuk mendapatkan 4 corner points

### Perspective Correction
1. Setelah deteksi dokumen, sistem akan otomatis koreksi perspektif
2. Transformasi perspective menggunakan homography
3. Hasil: dokumen tampak dari atas (bird's eye view)

### Image Enhancement
Pilihan filter yang tersedia:
- **Grayscale** - Konversi ke hitam putih
- **Adaptive Threshold** - Untuk dokumen teks yang jelas
- **Denoising** - Menghilangkan noise
- **Sharpening** - Mempertajam teks
- **Bilateral Filter** - Smoothing sambil mempertahankan edges

### Object Removal (Seam Carving)
1. Tandai area yang ingin dihapus (watermark, noda, dll)
2. Sistem akan menggunakan seam carving untuk menghapus secara content-aware
3. Gambar akan di-resize untuk menghapus objek tanpa distorsi berlebihan

### Real-time Tracking
1. Aktifkan webcam
2. Letakkan dokumen di depan kamera
3. Sistem akan track dokumen secara real-time
4. Tekan 's' untuk capture dan proses dokumen

## 🛠️ Teknologi yang Digunakan

- **OpenCV** - Computer vision operations
- **NumPy** - Numerical operations
- **Streamlit** - Web interface
- **Pillow** - Image processing
- **Scikit-image** - Advanced image processing

## 📊 Contoh Output

### Before & After
| Original | After Detection | After Enhancement |
|----------|----------------|-------------------|
| ![Original](docs/images/before.jpg) | ![Detected](docs/images/detected.jpg) | ![Enhanced](docs/images/after.jpg) |

## 🎓 Penjelasan Teknis

### Edge Detection
Menggunakan **Canny Edge Detection** dengan tahapan:
1. Gaussian blur untuk noise reduction
2. Gradient calculation (Sobel)
3. Non-maximum suppression
4. Double thresholding
5. Edge tracking by hysteresis

### Boundary Detection
1. **Contour Detection** - findContours untuk menemukan semua kontur
2. **Polygon Approximation** - Douglas-Peucker algorithm
3. **Shape Filtering** - Filter berdasarkan area dan jumlah vertices
4. **Corner Detection** - Ekstraksi 4 corner points

### Seam Carving Algorithm
1. **Energy Calculation** - Compute gradient magnitude
2. **Dynamic Programming** - Find minimum energy seam
3. **Seam Removal** - Remove pixels along seam
4. **Iteration** - Repeat until target size

## 🤝 Kontribusi

Proyek ini dibuat untuk Ujian Akhir Semester mata kuliah Visi Komputer.

## 📝 Lisensi

MIT License - Feel free to use for educational purposes

## 👨‍💻 Dibuat Oleh

Glenn Hakim
NPM: 2208107010072
Mata Kuliah: Visi Komputer
Semester: 7

---

**Note**: Pastikan untuk membaca `docs/usage_guide.md` untuk panduan penggunaan yang lebih detail.
