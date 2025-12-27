# 📖 Panduan Penggunaan Lengkap

## Daftar Isi
1. [Instalasi](#instalasi)
2. [Quick Start](#quick-start)
3. [Panduan Fitur](#panduan-fitur)
4. [API Reference](#api-reference)
5. [Troubleshooting](#troubleshooting)
6. [Best Practices](#best-practices)

---

## Instalasi

### Prasyarat
- Python 3.8 atau lebih baru
- pip (Python package manager)
- Webcam (opsional, untuk fitur real-time tracking)

### Langkah Instalasi

1. **Clone Repository**
```bash
git clone <repository-url>
cd UAS
```

2. **Buat Virtual Environment (Recommended)**
```bash
# Linux/Mac
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verifikasi Instalasi**
```bash
python -c "import cv2; print('OpenCV version:', cv2.__version__)"
```

---

## Quick Start

### 1. Menggunakan Streamlit Web App

Cara termudah untuk mencoba aplikasi:

```bash
streamlit run app.py
```

Browser akan otomatis terbuka di `http://localhost:8501`

### 2. Menggunakan Jupyter Notebook

```bash
jupyter notebook main.ipynb
```

Jalankan cell-cell secara berurutan untuk melihat demo lengkap.

### 3. Menggunakan Python Script

Buat file `test.py`:

```python
from modules import DocumentDetector, PerspectiveCorrector, ImageEnhancer
import cv2

# Load image
image = cv2.imread('sample_images/document1.jpg')

# Initialize modules
detector = DocumentDetector()
corrector = PerspectiveCorrector()
enhancer = ImageEnhancer()

# Process
corners = detector.detect_document(image)
if corners is not None:
    corrected = corrector.correct_perspective(image, corners)
    enhanced = enhancer.document_scan_effect(corrected)
    cv2.imwrite('output/result.jpg', enhanced)
    print("✅ Done!")
else:
    print("❌ Document not detected")
```

Jalankan:
```bash
python test.py
```

---

## Panduan Fitur

### 1. Auto Document Detection

**Fungsi**: Mendeteksi otomatis batas dokumen dari gambar

**Cara Penggunaan**:

```python
from modules.boundary_detector import DocumentDetector
import cv2

detector = DocumentDetector()
image = cv2.imread('your_image.jpg')

# Detect document corners
corners = detector.detect_document(image)

# Visualize detection
if corners is not None:
    vis = detector.visualize_detection(image, corners)
    cv2.imshow('Detection', vis)
    cv2.waitKey(0)
```

**Parameter Tuning**:

```python
# Untuk dokumen dengan edges yang kurang jelas
detector = DocumentDetector(
    canny_threshold1=30,  # Lower threshold
    canny_threshold2=100,  # Lower threshold
    blur_kernel_size=7     # More blur
)

# Untuk dokumen dengan edges yang sangat jelas
detector = DocumentDetector(
    canny_threshold1=75,   # Higher threshold
    canny_threshold2=200,  # Higher threshold
    blur_kernel_size=3     # Less blur
)
```

**Tips**:
- Pastikan dokumen memiliki kontras dengan background
- Dokumen harus memiliki tepi yang jelas
- Pencahayaan yang merata akan memberikan hasil terbaik
- Minimal area dokumen: 20% dari total gambar

---

### 2. Perspective Correction

**Fungsi**: Koreksi perspektif dokumen yang miring atau dilihat dari sudut

**Cara Penggunaan**:

```python
from modules.perspective_corrector import PerspectiveCorrector

corrector = PerspectiveCorrector()

# Basic correction
corrected = corrector.correct_perspective(image, corners)

# Auto correction dengan aspect ratio A4
corrected_a4 = corrector.auto_correct(image, corners, target_aspect_ratio=1.414)

# Auto deskew (koreksi kemiringan teks)
deskewed = corrector.deskew(image)
```

**Aspect Ratios Umum**:
- A4 Paper: 1.414 (210mm x 297mm)
- Letter: 1.294 (8.5" x 11")
- Legal: 1.647 (8.5" x 14")
- Square: 1.0

---

### 3. Image Enhancement

**Fungsi**: Meningkatkan kualitas gambar dengan berbagai filter

**Cara Penggunaan**:

```python
from modules.image_enhancer import ImageEnhancer

enhancer = ImageEnhancer()

# 1. Grayscale conversion
gray = enhancer.grayscale(image)

# 2. Adaptive thresholding (bagus untuk teks)
binary = enhancer.adaptive_threshold(image, block_size=11, c=2)

# 3. Bilateral filter (denoise sambil keep edges)
denoised = enhancer.denoise_bilateral(image, d=9, sigma_color=75, sigma_space=75)

# 4. CLAHE (contrast enhancement)
enhanced = enhancer.enhance_contrast(image, clip_limit=2.0)

# 5. Sharpen
sharpened = enhancer.sharpen(image, kernel_type='standard')

# 6. Complete scan effect (recommended)
scanned = enhancer.document_scan_effect(image)
```

**Kapan Menggunakan Apa**:

- **Adaptive Threshold**: Dokumen teks dengan pencahayaan tidak merata
- **Bilateral Filter**: Mengurangi noise tanpa blur teks
- **CLAHE**: Meningkatkan contrast untuk dokumen yang pudar
- **Sharpen**: Membuat teks lebih tajam
- **Document Scan Effect**: One-click solution untuk hasil terbaik

---

### 4. Seam Carving

**Fungsi**: Content-aware image resizing dan object removal

**Cara Penggunaan**:

```python
from modules.seam_carver import SeamCarver

carver = SeamCarver()

# Resize width (remove 50 pixels)
original_width = image.shape[1]
target_width = original_width - 50
resized = carver.resize_width(image, target_width)

# Remove object (requires binary mask)
# mask: 255 = object to remove, 0 = keep
mask = np.zeros(image.shape[:2], dtype=np.uint8)
mask[100:200, 100:300] = 255  # Mark area to remove
cleaned = carver.remove_object(image, mask, max_iterations=100)

# Visualize seam
energy = carver.calculate_energy(image)
seam = carver.find_vertical_seam(energy)
vis = carver.visualize_seam(image, seam)
```

**Warning**: Seam carving dapat lambat untuk gambar besar. Resize image terlebih dahulu jika perlu.

---

### 5. Real-time Document Tracking

**Fungsi**: Track dan capture dokumen dari webcam

**Cara Penggunaan**:

```python
from modules.document_tracker import DocumentTracker

tracker = DocumentTracker(camera_id=0)

# Define callback untuk auto capture
def on_capture(frame, corners):
    print("Document captured!")
    cv2.imwrite(f'capture_{time.time()}.jpg', frame)
    # Process captured document
    # ...

# Start tracking dengan auto capture
tracker.track_document(on_capture=on_capture, auto_capture=True)

# Atau manual capture (tekan 's' untuk capture)
tracker.track_document(auto_capture=False)

# Single capture dengan timeout
result = tracker.capture_single(timeout=30)
if result:
    frame, corners = result
    # Process...
```

**Controls**:
- **'s'**: Manual capture
- **'q'**: Quit tracking

**Tips untuk Tracking**:
- Gunakan pencahayaan yang baik
- Pegang dokumen steady untuk auto capture
- Stabilitas diukur dari pergerakan corners
- Auto capture terjadi setelah 10 frames stable

---

### 6. Batch Processing

**Cara Penggunaan**:

```python
from pathlib import Path

# Process multiple files
input_dir = Path('input_documents')
output_dir = Path('output_scanned')
output_dir.mkdir(exist_ok=True)

for img_path in input_dir.glob('*.jpg'):
    print(f"Processing {img_path.name}...")
    
    image = cv2.imread(str(img_path))
    corners = detector.detect_document(image)
    
    if corners is not None:
        corrected = corrector.correct_perspective(image, corners)
        enhanced = enhancer.document_scan_effect(corrected)
        
        output_path = output_dir / f"scanned_{img_path.name}"
        cv2.imwrite(str(output_path), enhanced)
        print(f"  ✓ Saved to {output_path}")
    else:
        print(f"  ✗ Failed to detect document")
```

---

## API Reference

### DocumentDetector

```python
class DocumentDetector:
    def __init__(self, canny_threshold1=50, canny_threshold2=150, blur_kernel_size=5)
    
    def detect_document(self, image, resize_width=800) -> np.ndarray | None
        """Returns 4 corner points atau None"""
    
    def visualize_detection(self, image, corners) -> np.ndarray
        """Returns image dengan visualization"""
    
    def get_edge_map(self, image, resize_width=800) -> np.ndarray
        """Returns binary edge map"""
```

### PerspectiveCorrector

```python
class PerspectiveCorrector:
    def correct_perspective(self, image, corners, output_size=None) -> np.ndarray
        """Returns perspective-corrected image"""
    
    def auto_correct(self, image, corners, target_aspect_ratio=None) -> np.ndarray
        """Returns corrected image dengan target aspect ratio"""
    
    def deskew(self, image) -> np.ndarray
        """Auto-deskew menggunakan Hough lines"""
    
    def correct_rotation(self, image, angle) -> np.ndarray
        """Rotate image by angle"""
```

### ImageEnhancer

```python
class ImageEnhancer:
    def adaptive_threshold(self, image, block_size=11, c=2, method='gaussian') -> np.ndarray
    def denoise_bilateral(self, image, d=9, sigma_color=75, sigma_space=75) -> np.ndarray
    def enhance_contrast(self, image, clip_limit=2.0) -> np.ndarray
    def sharpen(self, image, kernel_type='standard') -> np.ndarray
    def document_scan_effect(self, image) -> np.ndarray  # Recommended
```

### SeamCarver

```python
class SeamCarver:
    def resize_width(self, image, target_width) -> np.ndarray
    def remove_object(self, image, mask, max_iterations=None) -> np.ndarray
    def calculate_energy(self, image) -> np.ndarray
    def find_vertical_seam(self, energy) -> np.ndarray
```

### DocumentTracker

```python
class DocumentTracker:
    def __init__(self, camera_id=0)
    
    def track_document(self, on_capture=None, auto_capture=False, display_window='Document Tracker') -> bool
    
    def capture_single(self, timeout=30) -> tuple | None
    
    def get_camera_info(self) -> dict
```

---

## Troubleshooting

### Problem: Document tidak terdeteksi

**Solutions**:
1. Pastikan kontras antara dokumen dan background cukup
2. Coba turunkan threshold Canny:
   ```python
   detector = DocumentDetector(canny_threshold1=30, canny_threshold2=100)
   ```
3. Pastikan pencahayaan merata
4. Dokumen harus memenuhi minimal 20% area gambar

### Problem: Perspective correction tidak akurat

**Solutions**:
1. Pastikan 4 corners terdeteksi dengan benar
2. Periksa visualisasi detection terlebih dahulu
3. Manual adjust corners jika perlu:
   ```python
   corners = np.array([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], dtype=np.float32)
   ```

### Problem: Enhancement hasil terlalu gelap/terang

**Solutions**:
1. Adjust CLAHE clip_limit:
   ```python
   enhanced = enhancer.enhance_contrast(image, clip_limit=3.0)  # Higher = more contrast
   ```
2. Adjust adaptive threshold parameters:
   ```python
   binary = enhancer.adaptive_threshold(image, block_size=15, c=5)
   ```

### Problem: Seam carving terlalu lambat

**Solutions**:
1. Resize image terlebih dahulu:
   ```python
   from modules.utils import resize_image
   small = resize_image(image, max_dimension=800)
   carved = carver.resize_width(small, target_width)
   ```
2. Kurangi jumlah seams yang dihapus
3. Gunakan target_width yang lebih besar

### Problem: Webcam tidak terdeteksi

**Solutions**:
1. Cek camera ID (coba 0, 1, 2):
   ```python
   tracker = DocumentTracker(camera_id=1)
   ```
2. Pastikan webcam tidak digunakan aplikasi lain
3. Test webcam dengan:
   ```python
   cap = cv2.VideoCapture(0)
   print(cap.isOpened())
   ```

---

## Best Practices

### 1. Input Image Quality
- **Resolution**: Minimal 800x600 pixels
- **Format**: JPG, PNG (avoid heavy compression)
- **Lighting**: Even, avoid shadows
- **Background**: High contrast dengan dokumen

### 2. Processing Pipeline
```python
# Recommended pipeline
image = load_image('input.jpg')

# 1. Detect
corners = detector.detect_document(image)
if corners is None:
    # Fallback: manual processing
    pass

# 2. Correct
corrected = corrector.correct_perspective(image, corners)

# 3. Enhance
enhanced = enhancer.document_scan_effect(corrected)

# 4. Save
save_image(enhanced, 'output/result.jpg')
```

### 3. Performance Optimization
```python
# Resize large images sebelum processing
if image.shape[1] > 1920:
    image = resize_image(image, width=1920)

# Use lower resolution untuk detection
corners = detector.detect_document(image, resize_width=600)

# Process dengan original resolution
corrected = corrector.correct_perspective(image, corners)
```

### 4. Error Handling
```python
try:
    corners = detector.detect_document(image)
    if corners is None:
        raise ValueError("Document not detected")
    
    corrected = corrector.correct_perspective(image, corners)
    enhanced = enhancer.document_scan_effect(corrected)
    
except Exception as e:
    print(f"Error: {e}")
    # Fallback processing atau logging
```

---

## Contoh Use Cases

### Use Case 1: Simple Document Scanner
```python
def scan_document(input_path, output_path):
    image = cv2.imread(input_path)
    
    detector = DocumentDetector()
    corrector = PerspectiveCorrector()
    enhancer = ImageEnhancer()
    
    corners = detector.detect_document(image)
    if corners is not None:
        corrected = corrector.correct_perspective(image, corners)
        enhanced = enhancer.document_scan_effect(corrected)
        cv2.imwrite(output_path, enhanced)
        return True
    return False
```

### Use Case 2: Batch Processing dengan Progress
```python
from tqdm import tqdm

def batch_scan(input_dir, output_dir):
    input_files = list(Path(input_dir).glob('*.jpg'))
    
    for file in tqdm(input_files, desc="Scanning"):
        output = output_dir / f"scanned_{file.name}"
        scan_document(str(file), str(output))
```

### Use Case 3: Real-time Scanner dengan Auto Save
```python
def auto_scanner():
    tracker = DocumentTracker()
    counter = 0
    
    def save_scan(frame, corners):
        nonlocal counter
        corrected = corrector.correct_perspective(frame, corners)
        enhanced = enhancer.document_scan_effect(corrected)
        cv2.imwrite(f'scan_{counter:03d}.jpg', enhanced)
        counter += 1
        print(f"Saved scan_{counter:03d}.jpg")
    
    tracker.track_document(on_capture=save_scan, auto_capture=True)
```

---

**Untuk pertanyaan lebih lanjut, silakan buka issue di GitHub repository atau hubungi maintainer.**
