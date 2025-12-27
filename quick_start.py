"""
Quick start script untuk test semua fitur
"""

import cv2
import numpy as np
from pathlib import Path
from modules import (
    DocumentDetector,
    PerspectiveCorrector,
    ImageEnhancer,
    SeamCarver,
    DocumentTracker
)
from modules.utils import save_image


def create_sample_image():
    """Create sample document untuk testing"""
    img = np.ones((800, 600, 3), dtype=np.uint8) * 255
    
    cv2.putText(img, "SAMPLE DOCUMENT", (100, 100), 
                cv2.FONT_HERSHEY_BOLD, 1.5, (0, 0, 0), 3)
    
    texts = [
        "This is a test document for",
        "Smart Document Scanner",
        "",
        "Computer Vision Project",
        "Mata Kuliah: Visi Komputer"
    ]
    
    y = 200
    for text in texts:
        cv2.putText(img, text, (80, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        y += 60
    
    cv2.rectangle(img, (50, 50), (550, 750), (0, 0, 0), 3)
    
    # Add perspective distortion
    src = np.float32([[50, 50], [550, 50], [550, 750], [50, 750]])
    dst = np.float32([[150, 100], [650, 80], [680, 720], [100, 700]])
    M = cv2.getPerspectiveTransform(src, dst)
    distorted = cv2.warpPerspective(img, M, (800, 800), 
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(200, 200, 200))
    
    return distorted


def test_full_pipeline():
    """Test complete pipeline"""
    print("=" * 60)
    print("🚀 SMART DOCUMENT SCANNER - QUICK TEST")
    print("=" * 60)
    
    # Create directories
    Path('sample_images').mkdir(exist_ok=True)
    Path('output').mkdir(exist_ok=True)
    
    # Create sample image
    print("\n📝 Creating sample document...")
    sample = create_sample_image()
    save_image(sample, 'sample_images/test_doc.jpg')
    print("   ✓ Sample created: sample_images/test_doc.jpg")
    
    # Initialize modules
    print("\n🔧 Initializing modules...")
    detector = DocumentDetector()
    corrector = PerspectiveCorrector()
    enhancer = ImageEnhancer()
    print("   ✓ All modules initialized")
    
    # Step 1: Detect document
    print("\n🔍 Step 1: Detecting document boundaries...")
    corners = detector.detect_document(sample)
    
    if corners is None:
        print("   ✗ Failed to detect document!")
        return False
    
    print(f"   ✓ Document detected! Found 4 corners")
    
    # Save detection visualization
    detection_vis = detector.visualize_detection(sample, corners)
    save_image(detection_vis, 'output/1_detection.jpg')
    
    # Step 2: Correct perspective
    print("\n🔄 Step 2: Correcting perspective...")
    corrected = corrector.correct_perspective(sample, corners)
    save_image(corrected, 'output/2_corrected.jpg')
    print(f"   ✓ Perspective corrected: {corrected.shape[:2]}")
    
    # Step 3: Enhance
    print("\n✨ Step 3: Enhancing image...")
    
    # Multiple enhancement versions
    enhanced_bw = enhancer.document_scan_effect(corrected)
    save_image(enhanced_bw, 'output/3_enhanced_bw.jpg')
    
    enhanced_color = enhancer.denoise_bilateral(corrected)
    enhanced_color = cv2.cvtColor(
        enhancer.enhance_contrast(cv2.cvtColor(enhanced_color, cv2.COLOR_BGR2GRAY)),
        cv2.COLOR_GRAY2BGR
    )
    save_image(enhanced_color, 'output/4_enhanced_color.jpg')
    
    print("   ✓ Enhancement completed")
    
    # Step 4: Test seam carving (optional)
    print("\n✂️  Step 4: Testing seam carving...")
    try:
        carver = SeamCarver()
        original_width = corrected.shape[1]
        target_width = original_width - 50
        carved = carver.resize_width(corrected, target_width)
        save_image(carved, 'output/5_seam_carved.jpg')
        print(f"   ✓ Seam carving: {original_width}px → {target_width}px")
    except Exception as e:
        print(f"   ⚠ Seam carving skipped: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\n📁 Output files saved in 'output/' directory:")
    print("   - 1_detection.jpg      (Boundary detection)")
    print("   - 2_corrected.jpg      (Perspective corrected)")
    print("   - 3_enhanced_bw.jpg    (Enhanced B&W)")
    print("   - 4_enhanced_color.jpg (Enhanced Color)")
    print("   - 5_seam_carved.jpg    (Seam carving demo)")
    
    print("\n🎯 Next Steps:")
    print("   1. Run Streamlit app: streamlit run app.py")
    print("   2. Open Jupyter notebook: jupyter notebook main.ipynb")
    print("   3. Try webcam tracking (requires webcam)")
    
    return True


def test_webcam():
    """Test webcam tracking (optional)"""
    print("\n📹 Testing Webcam Tracking...")
    print("   This requires a webcam connected to your computer")
    
    try:
        tracker = DocumentTracker()
        info = tracker.get_camera_info()
        
        if info:
            print(f"   ✓ Camera detected!")
            print(f"     Resolution: {info['width']}x{info['height']}")
            print(f"     FPS: {info['fps']}")
            print("\n   To test tracking, run:")
            print("   python -c \"from modules import DocumentTracker; DocumentTracker().track_document()\"")
        else:
            print("   ⚠ No camera detected")
    except Exception as e:
        print(f"   ⚠ Camera test skipped: {e}")


if __name__ == "__main__":
    # Run full pipeline test
    success = test_full_pipeline()
    
    if success:
        # Optionally test webcam
        response = input("\n❓ Test webcam? (y/n): ").lower()
        if response == 'y':
            test_webcam()
    
    print("\n👋 Thank you for testing Smart Document Scanner!")
