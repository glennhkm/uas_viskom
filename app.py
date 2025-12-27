"""
Smart Document Scanner - Streamlit Web Application
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.boundary_detector import DocumentDetector
from modules.image_enhancer import ImageEnhancer
from modules.perspective_corrector import PerspectiveCorrector
from modules.seam_carver import SeamCarver


# Page config
st.set_page_config(
    page_title="Smart Document Scanner",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
def load_image_from_upload(uploaded_file):
    """Load image dari uploaded file"""
    image = Image.open(uploaded_file)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def display_image(image, caption=""):
    """Display image di Streamlit"""
    if len(image.shape) == 2:  # Grayscale
        st.image(image, caption=caption, use_container_width=True)
    else:  # Color
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                caption=caption, use_container_width=True)


def get_download_link(image, filename="result.png"):
    """Create download link untuk image"""
    is_success, buffer = cv2.imencode(".png", image)
    if is_success:
        return buffer.tobytes()
    return None


# Initialize session state
if 'processed_image' not in st.session_state:
    st.session_state.processed_image = None
if 'original_image' not in st.session_state:
    st.session_state.original_image = None
if 'detected_corners' not in st.session_state:
    st.session_state.detected_corners = None


# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">📄 Smart Document Scanner</h1>', 
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Pemindai Dokumen Pintar dengan Auto Enhancement</p>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        mode = st.selectbox(
            "Pilih Mode",
            ["Auto Scan", "Manual Enhancement", "Object Removal", "Batch Processing"]
        )
        
        st.markdown("---")
        
        st.markdown("""
        ### 📋 Fitur:
        - ✅ Auto Document Detection
        - ✅ Perspective Correction
        - ✅ Image Enhancement
        - ✅ Object Removal
        - ✅ Batch Processing
        
        ### 📚 Materi CV:
        - Edge Detection (Canny)
        - Boundary Detection
        - Filtering
        - Seam Carving
        - Perspective Transform
        """)
    
    # Main content
    if mode == "Auto Scan":
        auto_scan_mode()
    elif mode == "Manual Enhancement":
        manual_enhancement_mode()
    elif mode == "Object Removal":
        object_removal_mode()
    else:
        batch_processing_mode()


def auto_scan_mode():
    """Mode auto scan - deteksi dan koreksi otomatis"""
    st.header("🚀 Auto Scan Mode")
    st.write("Upload gambar dokumen, sistem akan otomatis mendeteksi, koreksi, dan enhance.")
    
    uploaded_file = st.file_uploader("Upload Gambar Dokumen", 
                                     type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        try:
            # Load image
            image = load_image_from_upload(uploaded_file)
            st.session_state.original_image = image
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📥 Original")
                display_image(image, caption="Original Image")
            
            with st.spinner("🔍 Processing..."):
                # Initialize modules
                detector = DocumentDetector()
                corrector = PerspectiveCorrector()
                enhancer = ImageEnhancer()
                
                # Step 1: Detect document
                corners = detector.detect_document(image)
            
                if corners is not None:
                    st.session_state.detected_corners = corners
                    
                    # Visualize detection
                    detection_vis = detector.visualize_detection(image, corners)
                    
                    with col2:
                        st.subheader("🔍 Detection")
                        display_image(detection_vis, caption="Detected Corners")
                    
                    st.success(f"✅ Document detected! Corners: {len(corners)} points")
                    
                    # Step 2: Perspective correction
                    with st.spinner("📐 Correcting perspective..."):
                        corrected = corrector.correct_perspective(image, corners)
                    
                    # Step 3: Enhancement
                    with st.spinner("✨ Enhancing image..."):
                        enhanced = enhancer.document_scan_effect(corrected)
                    
                    st.session_state.processed_image = enhanced
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("✨ Results")
                    
                    col3, col4, col5 = st.columns(3)
                    
                    with col3:
                        st.write("**Perspective Corrected**")
                        display_image(corrected, caption="Corrected")
                    
                    with col4:
                        st.write("**Enhanced (B&W)**")
                        display_image(enhanced, caption="Black & White")
                    
                    with col5:
                        # Color enhanced version
                        with st.spinner("Processing color..."):
                            color_enhanced = enhancer.denoise_bilateral(corrected)
                            color_enhanced = cv2.cvtColor(
                                enhancer.enhance_contrast(
                                    cv2.cvtColor(color_enhanced, cv2.COLOR_BGR2GRAY)
                                ), cv2.COLOR_GRAY2BGR
                            )
                        st.write("**Enhanced (Color)**")
                        display_image(color_enhanced, caption="Color Enhanced")
                    
                    # Download options
                    st.markdown("---")
                    st.subheader("💾 Download")
                    
                    col6, col7, col8 = st.columns(3)
                    
                    with col6:
                        if st.download_button(
                            "Download Corrected",
                            data=get_download_link(corrected, "corrected.png"),
                            file_name="document_corrected.png",
                            mime="image/png"
                        ):
                            st.success("Downloaded!")
                    
                    with col7:
                        if st.download_button(
                            "Download Enhanced (B&W)",
                            data=get_download_link(enhanced, "enhanced_bw.png"),
                            file_name="document_enhanced_bw.png",
                            mime="image/png"
                        ):
                            st.success("Downloaded!")
                    
                    with col8:
                        if st.download_button(
                            "Download Enhanced (Color)",
                            data=get_download_link(color_enhanced, "enhanced_color.png"),
                            file_name="document_enhanced_color.png",
                            mime="image/png"
                        ):
                            st.success("Downloaded!")
                
                else:
                    with col2:
                        st.error("❌ Dokumen tidak terdeteksi!")
                        st.warning("""**Tips untuk detection yang lebih baik:**
                        - Pastikan dokumen terlihat jelas
                        - Gunakan latar belakang yang kontras
                        - Hindari bayangan yang terlalu gelap
                        - Pastikan dokumen tidak terlalu kecil dalam frame
                        """)
                        
                        # Show edge map for debugging
                        edge_map = detector.get_edge_map(image)
                        if edge_map is not None:
                            st.write("**Edge Detection Map (Debug):**")
                            display_image(edge_map, caption="Edges Detected")
        
        except Exception as e:
            st.error(f"❌ Error loading image: {str(e)}")
            import traceback
            st.code(traceback.format_exc())


def manual_enhancement_mode():
    """Mode manual enhancement dengan kontrol filter"""
    st.header("🎨 Manual Enhancement Mode")
    st.write("Kontrol manual untuk enhancement dengan berbagai filter.")
    
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        image = load_image_from_upload(uploaded_file)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            display_image(image)
        
        # Enhancement options
        st.sidebar.header("Enhancement Options")
        
        enhancement_type = st.sidebar.selectbox(
            "Filter Type",
            ["Grayscale", "Adaptive Threshold", "Bilateral Filter", 
             "CLAHE", "Sharpen", "Denoise", "Auto Scan Effect"]
        )
        
        enhancer = ImageEnhancer()
        result = image.copy()
        
        if enhancement_type == "Grayscale":
            result = enhancer.grayscale(image)
        
        elif enhancement_type == "Adaptive Threshold":
            block_size = st.sidebar.slider("Block Size", 3, 51, 11, step=2)
            c = st.sidebar.slider("C Value", 0, 20, 2)
            result = enhancer.adaptive_threshold(image, block_size, c)
        
        elif enhancement_type == "Bilateral Filter":
            d = st.sidebar.slider("Diameter", 3, 15, 9)
            sigma_color = st.sidebar.slider("Sigma Color", 10, 150, 75)
            sigma_space = st.sidebar.slider("Sigma Space", 10, 150, 75)
            result = enhancer.denoise_bilateral(image, d, sigma_color, sigma_space)
        
        elif enhancement_type == "CLAHE":
            clip_limit = st.sidebar.slider("Clip Limit", 1.0, 5.0, 2.0, 0.1)
            result = enhancer.enhance_contrast(image, clip_limit)
        
        elif enhancement_type == "Sharpen":
            kernel_type = st.sidebar.selectbox("Kernel", ["standard", "strong", "unsharp"])
            result = enhancer.sharpen(image, kernel_type)
        
        elif enhancement_type == "Denoise":
            h = st.sidebar.slider("Filter Strength", 5, 20, 10)
            result = enhancer.denoise_nlmeans(image, h)
        
        else:  # Auto Scan Effect
            result = enhancer.document_scan_effect(image)
        
        with col2:
            st.subheader("Enhanced")
            display_image(result)
        
        # Download
        st.download_button(
            "💾 Download Result",
            data=get_download_link(result),
            file_name="enhanced.png",
            mime="image/png"
        )


def object_removal_mode():
    """Mode object removal dengan seam carving"""
    st.header("🗑️ Object Removal Mode")
    st.write("Hapus watermark atau objek tidak diinginkan dengan seam carving.")
    
    uploaded_file = st.file_uploader("Upload Gambar", type=['jpg', 'jpeg', 'png', 'bmp'])
    
    if uploaded_file:
        image = load_image_from_upload(uploaded_file)
        
        st.info("⚠️ Fitur ini memerlukan mask untuk area yang akan dihapus. " + 
                "Dalam implementasi lengkap, Anda bisa menggambar mask secara interaktif.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original")
            display_image(image)
        
        # Resize option
        resize_width = st.sidebar.slider(
            "Resize Width (seam carving)",
            100, image.shape[1], image.shape[1] - 50
        )
        
        if st.button("Apply Seam Carving Resize"):
            with st.spinner("Processing..."):
                carver = SeamCarver()
                result = carver.resize_width(image, resize_width)
                
                with col2:
                    st.subheader(f"Resized to {resize_width}px")
                    display_image(result)
                
                st.download_button(
                    "💾 Download Result",
                    data=get_download_link(result),
                    file_name="seam_carved.png",
                    mime="image/png"
                )


def batch_processing_mode():
    """Mode batch processing untuk banyak dokumen"""
    st.header("📚 Batch Processing Mode")
    st.write("Proses banyak dokumen sekaligus.")
    
    uploaded_files = st.file_uploader(
        "Upload Multiple Images",
        type=['jpg', 'jpeg', 'png', 'bmp'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"📊 Total files: {len(uploaded_files)}")
        
        if st.button("🚀 Process All"):
            detector = DocumentDetector()
            corrector = PerspectiveCorrector()
            enhancer = ImageEnhancer()
            
            progress_bar = st.progress(0)
            results = []
            
            for idx, file in enumerate(uploaded_files):
                image = load_image_from_upload(file)
                
                # Process
                corners = detector.detect_document(image)
                if corners is not None:
                    corrected = corrector.correct_perspective(image, corners)
                    enhanced = enhancer.document_scan_effect(corrected)
                    results.append((file.name, enhanced))
                
                progress_bar.progress((idx + 1) / len(uploaded_files))
            
            st.success(f"✅ Processed {len(results)}/{len(uploaded_files)} images successfully!")
            
            # Display results
            cols = st.columns(3)
            for idx, (name, img) in enumerate(results):
                with cols[idx % 3]:
                    st.write(f"**{name}**")
                    display_image(img)
                    st.download_button(
                        f"Download",
                        data=get_download_link(img),
                        file_name=f"processed_{name}",
                        mime="image/png",
                        key=f"download_{idx}"
                    )


if __name__ == "__main__":
    main()
