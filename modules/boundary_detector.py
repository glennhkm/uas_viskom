"""
Document Detector - Boundary Detection Module
Menggunakan Edge Detection dan Contour Detection untuk menemukan batas dokumen
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List
from .utils import order_points, resize_image, convert_to_grayscale


class DocumentDetector:
    """
    Class untuk mendeteksi batas dokumen dalam gambar
    
    Menggunakan:
    - Canny Edge Detection
    - Contour Detection
    - Polygon Approximation
    """
    
    def __init__(self, canny_threshold1: int = 30, canny_threshold2: int = 100,
                 blur_kernel_size: int = 5):
        """
        Initialize DocumentDetector
        
        Args:
            canny_threshold1: Lower threshold untuk Canny (lowered for better detection)
            canny_threshold2: Upper threshold untuk Canny (lowered for better detection)
            blur_kernel_size: Kernel size untuk Gaussian blur
        """
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.blur_kernel_size = blur_kernel_size
    
    def preprocess(self, image: np.ndarray, enhance: bool = True) -> np.ndarray:
        """
        Preprocessing image sebelum edge detection
        
        Args:
            image: Input image (BGR)
            enhance: Apply adaptive enhancement for low-contrast images
            
        Returns:
            numpy.ndarray: Preprocessed grayscale image
        """
        # Convert ke grayscale
        gray = convert_to_grayscale(image)
        
        if enhance:
            # Apply CLAHE untuk enhance contrast
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            
            # Adaptive threshold untuk highlight edges
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
            # Combine dengan original
            gray = cv2.addWeighted(gray, 0.7, binary, 0.3, 0)
        
        # Gaussian blur untuk noise reduction
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        return blurred
    
    def detect_edges(self, image: np.ndarray, auto_threshold: bool = True) -> np.ndarray:
        """
        Deteksi edges menggunakan Canny Edge Detection
        
        Canny Edge Detection steps:
        1. Noise reduction dengan Gaussian filter
        2. Gradient calculation
        3. Non-maximum suppression
        4. Double thresholding
        5. Edge tracking by hysteresis
        
        Args:
            image: Input grayscale image
            auto_threshold: Use automatic threshold calculation
            
        Returns:
            numpy.ndarray: Binary edge map
        """
        if auto_threshold:
            # Automatic threshold using median
            median = np.median(image)
            lower = int(max(0, 0.5 * median))
            upper = int(min(255, 1.5 * median))
            edges = cv2.Canny(image, lower, upper)
        else:
            edges = cv2.Canny(image, self.canny_threshold1, self.canny_threshold2)
        
        # Dilasi untuk menghubungkan edges yang terputus
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)
        
        # Close small gaps
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        return edges
    
    def find_document_contour(self, edges: np.ndarray, 
                             original_shape: Tuple[int, int],
                             min_area_ratio: float = 0.1,
                             max_area_ratio: float = 0.95) -> Optional[np.ndarray]:
        """
        Temukan contour dokumen dari edge map
        
        Args:
            edges: Binary edge map
            original_shape: Shape dari original image
            min_area_ratio: Minimum area ratio (default 0.1 = 10%)
            max_area_ratio: Maximum area ratio (default 0.95 = 95% - skip frame)
            
        Returns:
            numpy.ndarray: Contour points (4 corners) atau None
        """
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Sort contours berdasarkan area (descending)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        image_area = original_shape[0] * original_shape[1]
        height, width = original_shape
        
        # Cari contour dengan 4 corners
        for contour in contours[:10]:  # Check top 10 largest contours
            # Calculate perimeter
            peri = cv2.arcLength(contour, True)
            
            # Try multiple epsilon values untuk polygon approximation
            for epsilon_factor in [0.02, 0.03, 0.04, 0.05]:
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                # Jika contour memiliki 4 points, kemungkinan ini dokumen
                if len(approx) == 4:
                    # Validasi: area harus cukup besar tapi tidak terlalu besar
                    area = cv2.contourArea(approx)
                    area_ratio = area / image_area
                    
                    # Skip jika terlalu kecil atau terlalu besar (kemungkinan outer frame)
                    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
                        continue
                    
                    # Additional validation: check if it's roughly rectangular
                    corners = approx.reshape(4, 2)
                    if not self._is_valid_rectangle(corners):
                        continue
                    
                    # CRITICAL: Skip jika terlalu dekat dengan image border
                    # (kemungkinan besar ini outer frame, bukan dokumen)
                    # Relax: hanya skip jika semua corners dekat border
                    if self._is_near_border(corners, width, height, margin_ratio=0.05):
                        continue
                    
                    # Additional check: Pastikan bukan perfect rectangle di edge
                    # (outer frame biasanya sangat perfect align dengan image boundary)
                    if self._is_likely_outer_frame(corners, width, height):
                        continue
                    
                    # Jika lolos semua validasi, ini kemungkinan dokumen yang benar
                    return corners
        
        return None
    
    def _is_near_border(self, corners: np.ndarray, width: int, height: int, margin_ratio: float = 0.05) -> bool:
        """
        Check jika corners terlalu dekat dengan border image
        
        Args:
            corners: 4 corner points
            width: Image width
            height: Image height
            margin_ratio: Minimum distance from border as ratio of dimension (default 0.05 = 5%)
            
        Returns:
            bool: True if any corner is too close to border
        """
        margin_x = int(width * margin_ratio)
        margin_y = int(height * margin_ratio)
        
        # Count berapa corners yang dekat border
        near_border_count = 0
        for corner in corners:
            x, y = corner
            # Check jika dekat dengan salah satu border
            if (x < margin_x or x > width - margin_x or 
                y < margin_y or y > height - margin_y):
                near_border_count += 1
        
        # Hanya reject jika SEMUA 4 corners dekat border (outer frame)
        # Jika hanya 2-3 corners, masih mungkin dokumen valid yang miring
        return near_border_count == 4
    
    def _is_likely_outer_frame(self, corners: np.ndarray, width: int, height: int) -> bool:
        """
        Check jika contour kemungkinan adalah outer frame (bukan dokumen)
        
        Outer frame characteristics:
        - Corners sangat dekat dengan image edges (aligned)
        - Aspect ratio sangat mirip dengan image aspect ratio
        - Area ratio > 90%
        
        Args:
            corners: 4 corner points
            width: Image width
            height: Image height
            
        Returns:
            bool: True if likely an outer frame
        """
        ordered = order_points(corners)
        
        # Calculate bounding box
        min_x = np.min(ordered[:, 0])
        max_x = np.max(ordered[:, 0])
        min_y = np.min(ordered[:, 1])
        max_y = np.max(ordered[:, 1])
        
        doc_width = max_x - min_x
        doc_height = max_y - min_y
        
        # Check if aspect ratio sangat mirip dengan image
        image_aspect = width / height
        doc_aspect = doc_width / doc_height
        aspect_diff = abs(image_aspect - doc_aspect)
        
        # Jika aspect ratio hampir sama DAN area > 90%, kemungkinan outer frame
        area_ratio = cv2.contourArea(corners) / (width * height)
        
        if aspect_diff < 0.1 and area_ratio > 0.90:
            return True
        
        # Check jika corners aligned dengan image boundaries (within 3%)
        margin = 0.03
        aligned_count = 0
        
        for x, y in ordered:
            if (x < width * margin or x > width * (1 - margin) or
                y < height * margin or y > height * (1 - margin)):
                aligned_count += 1
        
        # Jika semua 4 corners aligned dengan boundary, ini outer frame
        return aligned_count >= 4
    
    def _find_document_relaxed(self, edges: np.ndarray, original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Fallback method dengan validasi yang lebih relax
        Hanya check: 4 corners, area reasonable, NOT outer frame
        
        Args:
            edges: Binary edge map
            original_shape: Shape dari original image
            
        Returns:
            numpy.ndarray: Contour points atau None
        """
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        image_area = original_shape[0] * original_shape[1]
        height, width = original_shape
        
        for contour in contours[:15]:  # Check more contours
            peri = cv2.arcLength(contour, True)
            
            # Try multiple epsilon values
            for epsilon_factor in [0.02, 0.03, 0.04, 0.05, 0.06]:
                approx = cv2.approxPolyDP(contour, epsilon_factor * peri, True)
                
                if len(approx) == 4:
                    area = cv2.contourArea(approx)
                    area_ratio = area / image_area
                    
                    # Hanya basic checks
                    if area_ratio < 0.05 or area_ratio > 0.95:
                        continue
                    
                    corners = approx.reshape(4, 2)
                    
                    # Skip HANYA jika jelas outer frame (semua corners di edge)
                    if self._is_likely_outer_frame(corners, width, height):
                        continue
                    
                    # Accept!
                    return corners
        
        return None
    
    def _is_valid_rectangle(self, corners: np.ndarray) -> bool:
        """
        Validate if 4 corners form a valid rectangle
        
        Args:
            corners: 4 corner points
            
        Returns:
            bool: True if valid rectangle
        """
        # Check if corners are roughly forming a rectangle
        # Calculate angles between edges
        ordered = order_points(corners)
        
        # Calculate edge vectors
        edges = []
        for i in range(4):
            next_i = (i + 1) % 4
            edge = ordered[next_i] - ordered[i]
            edges.append(edge)
        
        # Check if angles are roughly 90 degrees (±45 degrees tolerance)
        for i in range(4):
            next_i = (i + 1) % 4
            dot = np.dot(edges[i], edges[next_i])
            mag1 = np.linalg.norm(edges[i])
            mag2 = np.linalg.norm(edges[next_i])
            
            if mag1 == 0 or mag2 == 0:
                return False
            
            cos_angle = dot / (mag1 * mag2)
            # cos(90°) = 0, increase tolerance untuk dokumen yang agak miring
            # 0.85 ≈ cos(32°), jadi allow angle 58° - 122°
            if abs(cos_angle) > 0.85:  # Not roughly perpendicular
                return False
        
        return True
    
    def detect_document(self, image: np.ndarray, 
                       resize_width: int = 800) -> Optional[np.ndarray]:
        """
        Main method untuk mendeteksi dokumen
        
        Args:
            image: Input image (BGR) atau path ke image
            resize_width: Width untuk resize (untuk processing lebih cepat)
            
        Returns:
            numpy.ndarray: 4 corner points dari dokumen atau None
        """
        # Jika input adalah string (path), load image
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Simpan original size untuk scaling kembali
        original_height, original_width = image.shape[:2]
        
        # Resize untuk processing lebih cepat
        resized = resize_image(image, width=resize_width)
        resized_height, resized_width = resized.shape[:2]
        
        # Ratio untuk scaling kembali ke original size
        ratio_width = original_width / resized_width
        ratio_height = original_height / resized_height
        
        # Preprocessing dengan enhancement
        preprocessed = self.preprocess(resized, enhance=True)
        
        # Edge detection dengan auto threshold
        edges = self.detect_edges(preprocessed, auto_threshold=True)
        
        # Find document contour dengan threshold rendah dan skip outer frame
        contour = self.find_document_contour(edges, resized.shape[:2], 
                                             min_area_ratio=0.1, max_area_ratio=0.95)
        
        # Fallback 1: try without enhancement jika gagal
        if contour is None:
            preprocessed = self.preprocess(resized, enhance=False)
            edges = self.detect_edges(preprocessed, auto_threshold=False)
            contour = self.find_document_contour(edges, resized.shape[:2], 
                                                 min_area_ratio=0.05, max_area_ratio=0.95)
        
        # Fallback 2: try dengan validasi yang lebih relax
        if contour is None:
            # Cari contour tanpa strict validation
            contour = self._find_document_relaxed(edges, resized.shape[:2])
        
        if contour is None:
            return None
        
        # Scale kembali ke original size
        contour = contour.astype(np.float32)
        contour[:, 0] *= ratio_width
        contour[:, 1] *= ratio_height
        
        # Order points
        ordered = order_points(contour)
        
        return ordered
    
    def visualize_detection(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """
        Visualisasi hasil deteksi dengan menggambar corners dan lines
        
        Args:
            image: Original image
            corners: 4 corner points
            
        Returns:
            numpy.ndarray: Image dengan visualization
        """
        output = image.copy()
        
        # Draw contour lines
        pts = corners.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(output, [pts], True, (0, 255, 0), 3)
        
        # Draw corner points
        for i, point in enumerate(corners):
            x, y = int(point[0]), int(point[1])
            cv2.circle(output, (x, y), 10, (0, 0, 255), -1)
            # Label corners
            labels = ['TL', 'TR', 'BR', 'BL']
            cv2.putText(output, labels[i], (x - 20, y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return output
    
    def get_edge_map(self, image: np.ndarray, resize_width: int = 800) -> np.ndarray:
        """
        Get edge map untuk debugging/visualization
        
        Args:
            image: Input image
            resize_width: Width untuk resize
            
        Returns:
            numpy.ndarray: Edge map
        """
        resized = resize_image(image, width=resize_width)
        preprocessed = self.preprocess(resized)
        edges = self.detect_edges(preprocessed)
        return edges
