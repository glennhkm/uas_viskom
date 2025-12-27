"""
Perspective Corrector Module
Koreksi perspektif dokumen yang miring atau dilihat dari sudut
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .utils import order_points, calculate_distance


class PerspectiveCorrector:
    """
    Class untuk koreksi perspektif dokumen
    
    Menggunakan:
    - Perspective transformation (Homography)
    - 4-point transformation
    """
    
    def __init__(self):
        """Initialize PerspectiveCorrector"""
        pass
    
    def calculate_new_dimensions(self, corners: np.ndarray) -> Tuple[int, int]:
        """
        Calculate dimensi output setelah perspective correction
        
        Menggunakan maximum width dan height dari keempat sisi
        
        Args:
            corners: Ordered corner points [TL, TR, BR, BL]
            
        Returns:
            tuple: (width, height)
        """
        # Unpack points
        (tl, tr, br, bl) = corners
        
        # Calculate width
        # Distance antara top-left dan top-right
        width_top = calculate_distance((tl[0], tl[1]), (tr[0], tr[1]))
        # Distance antara bottom-left dan bottom-right
        width_bottom = calculate_distance((bl[0], bl[1]), (br[0], br[1]))
        # Ambil yang maksimum
        max_width = max(int(width_top), int(width_bottom))
        
        # Calculate height
        # Distance antara top-left dan bottom-left
        height_left = calculate_distance((tl[0], tl[1]), (bl[0], bl[1]))
        # Distance antara top-right dan bottom-right
        height_right = calculate_distance((tr[0], tr[1]), (br[0], br[1]))
        # Ambil yang maksimum
        max_height = max(int(height_left), int(height_right))
        
        return (max_width, max_height)
    
    def correct_perspective(self, image: np.ndarray, 
                          corners: np.ndarray,
                          output_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Main method untuk koreksi perspektif
        
        Process:
        1. Order corner points
        2. Calculate output dimensions
        3. Define destination points
        4. Calculate perspective transform matrix (homography)
        5. Apply warp perspective
        
        Args:
            image: Input image
            corners: 4 corner points
            output_size: Optional output size (width, height)
            
        Returns:
            numpy.ndarray: Perspective corrected image
        """
        # Order points
        ordered_corners = order_points(corners)
        
        # Calculate output dimensions
        if output_size is None:
            width, height = self.calculate_new_dimensions(ordered_corners)
        else:
            width, height = output_size
        
        # Define destination points (rectangle)
        dst = np.array([
            [0, 0],                    # Top-left
            [width - 1, 0],            # Top-right
            [width - 1, height - 1],   # Bottom-right
            [0, height - 1]            # Bottom-left
        ], dtype="float32")
        
        # Calculate perspective transform matrix
        # Ini adalah homography matrix yang memetakan source points ke destination
        M = cv2.getPerspectiveTransform(ordered_corners, dst)
        
        # Apply perspective transformation
        warped = cv2.warpPerspective(image, M, (width, height))
        
        return warped
    
    def auto_correct(self, image: np.ndarray, 
                    corners: np.ndarray,
                    target_aspect_ratio: Optional[float] = None) -> np.ndarray:
        """
        Auto correct dengan aspect ratio tertentu
        
        Args:
            image: Input image
            corners: 4 corner points
            target_aspect_ratio: Target aspect ratio (width/height)
                                Common: A4 = 1.414, Letter = 1.294
            
        Returns:
            numpy.ndarray: Corrected image
        """
        # Order points
        ordered_corners = order_points(corners)
        
        # Calculate natural dimensions
        width, height = self.calculate_new_dimensions(ordered_corners)
        
        # Adjust berdasarkan target aspect ratio jika ada
        if target_aspect_ratio:
            current_ratio = width / height
            
            if current_ratio > target_aspect_ratio:
                # Width terlalu besar, adjust width
                width = int(height * target_aspect_ratio)
            else:
                # Height terlalu besar, adjust height
                height = int(width / target_aspect_ratio)
        
        return self.correct_perspective(image, corners, (width, height))
    
    def correct_rotation(self, image: np.ndarray, angle: float) -> np.ndarray:
        """
        Koreksi rotasi sederhana
        
        Args:
            image: Input image
            angle: Rotation angle dalam derajat (positive = counter-clockwise)
            
        Returns:
            numpy.ndarray: Rotated image
        """
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions to prevent cropping
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_width = int((height * sin) + (width * cos))
        new_height = int((height * cos) + (width * sin))
        
        # Adjust translation
        M[0, 2] += (new_width / 2) - center[0]
        M[1, 2] += (new_height / 2) - center[1]
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (new_width, new_height),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255))
        
        return rotated
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """
        Auto deskew (koreksi kemiringan teks)
        
        Menggunakan Hough Line Transform untuk detect skew angle
        
        Args:
            image: Input image (grayscale atau color)
            
        Returns:
            numpy.ndarray: Deskewed image
        """
        # Convert ke grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Threshold
        thresh = cv2.threshold(gray, 0, 255, 
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect lines menggunakan HoughLinesP
        lines = cv2.HoughLinesP(thresh, 1, np.pi/180, 100,
                               minLineLength=100, maxLineGap=10)
        
        if lines is None:
            return image
        
        # Calculate angles
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi
            angles.append(angle)
        
        # Get median angle
        median_angle = np.median(angles)
        
        # Koreksi jika angle > 45 derajat
        if median_angle < -45:
            median_angle = 90 + median_angle
        elif median_angle > 45:
            median_angle = median_angle - 90
        
        # Rotate image
        if abs(median_angle) > 0.5:  # Only rotate if angle significant
            return self.correct_rotation(image, median_angle)
        
        return image
    
    def crop_and_resize(self, image: np.ndarray, 
                       target_size: Tuple[int, int],
                       maintain_aspect: bool = True) -> np.ndarray:
        """
        Crop dan resize ke target size
        
        Args:
            image: Input image
            target_size: Target size (width, height)
            maintain_aspect: Maintain aspect ratio atau tidak
            
        Returns:
            numpy.ndarray: Resized image
        """
        if maintain_aspect:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            # Calculate aspect ratios
            img_ratio = w / h
            target_ratio = target_w / target_h
            
            if img_ratio > target_ratio:
                # Image lebih lebar, crop width
                new_w = int(h * target_ratio)
                start_x = (w - new_w) // 2
                cropped = image[:, start_x:start_x + new_w]
            else:
                # Image lebih tinggi, crop height
                new_h = int(w / target_ratio)
                start_y = (h - new_h) // 2
                cropped = image[start_y:start_y + new_h, :]
            
            # Resize
            resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        else:
            resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        
        return resized
