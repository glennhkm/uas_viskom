"""
Image Enhancer - Filtering Module
Berbagai filter untuk meningkatkan kualitas dokumen
"""

import cv2
import numpy as np
from typing import Tuple, Optional
from .utils import convert_to_grayscale


class ImageEnhancer:
    """
    Class untuk enhancement dan filtering image dokumen
    
    Termasuk:
    - Adaptive Thresholding
    - Denoising filters
    - Sharpening
    - Bilateral filter
    - Contrast enhancement
    """
    
    def __init__(self):
        """Initialize ImageEnhancer"""
        pass
    
    def grayscale(self, image: np.ndarray) -> np.ndarray:
        """
        Convert ke grayscale
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Grayscale image
        """
        return convert_to_grayscale(image)
    
    def adaptive_threshold(self, image: np.ndarray, 
                          block_size: int = 11, 
                          c: int = 2,
                          method: str = 'gaussian') -> np.ndarray:
        """
        Adaptive thresholding untuk dokumen dengan pencahayaan tidak merata
        
        Algoritma:
        - Threshold value dihitung untuk setiap pixel berdasarkan neighborhood
        - Bagus untuk dokumen dengan shadow atau pencahayaan tidak rata
        
        Args:
            image: Input image (akan di-convert ke grayscale)
            block_size: Size of neighborhood area (harus ganjil)
            c: Constant yang dikurangi dari weighted mean
            method: 'gaussian' atau 'mean'
            
        Returns:
            numpy.ndarray: Binary image
        """
        # Convert ke grayscale jika perlu
        gray = convert_to_grayscale(image)
        
        # Ensure block_size is odd
        if block_size % 2 == 0:
            block_size += 1
        
        if method == 'gaussian':
            adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        else:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
        
        binary = cv2.adaptiveThreshold(
            gray, 255, adaptive_method,
            cv2.THRESH_BINARY, block_size, c
        )
        
        return binary
    
    def denoise_bilateral(self, image: np.ndarray, 
                         d: int = 9, 
                         sigma_color: int = 75, 
                         sigma_space: int = 75) -> np.ndarray:
        """
        Bilateral filter - smoothing sambil mempertahankan edges
        
        Keuntungan:
        - Mengurangi noise
        - Mempertahankan edges yang sharp
        - Tidak membuat teks blur
        
        Args:
            image: Input image
            d: Diameter of pixel neighborhood
            sigma_color: Filter sigma in color space
            sigma_space: Filter sigma in coordinate space
            
        Returns:
            numpy.ndarray: Denoised image
        """
        return cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    
    def denoise_nlmeans(self, image: np.ndarray, 
                       h: int = 10, 
                       template_window_size: int = 7,
                       search_window_size: int = 21) -> np.ndarray:
        """
        Non-local means denoising - untuk noise yang lebih berat
        
        Args:
            image: Input image
            h: Filter strength (higher = lebih smooth)
            template_window_size: Size of template patch
            search_window_size: Size of search area
            
        Returns:
            numpy.ndarray: Denoised image
        """
        if len(image.shape) == 2:  # Grayscale
            return cv2.fastNlMeansDenoising(image, None, h, 
                                           template_window_size, 
                                           search_window_size)
        else:  # Color
            return cv2.fastNlMeansDenoisingColored(image, None, h, h,
                                                   template_window_size,
                                                   search_window_size)
    
    def sharpen(self, image: np.ndarray, kernel_type: str = 'standard') -> np.ndarray:
        """
        Sharpen image untuk membuat teks lebih jelas
        
        Args:
            image: Input image
            kernel_type: 'standard', 'strong', atau 'unsharp'
            
        Returns:
            numpy.ndarray: Sharpened image
        """
        if kernel_type == 'standard':
            kernel = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])
        elif kernel_type == 'strong':
            kernel = np.array([[-1, -1, -1],
                             [-1, 9, -1],
                             [-1, -1, -1]])
        else:  # unsharp masking
            gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
            sharpened = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
            return sharpened
        
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def enhance_contrast(self, image: np.ndarray, 
                        clip_limit: float = 2.0,
                        tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
        """
        CLAHE - Contrast Limited Adaptive Histogram Equalization
        
        Meningkatkan contrast dengan histogram equalization yang adaptive
        
        Args:
            image: Input image
            clip_limit: Threshold for contrast limiting
            tile_grid_size: Size of grid for histogram equalization
            
        Returns:
            numpy.ndarray: Contrast enhanced image
        """
        gray = convert_to_grayscale(image)
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced = clahe.apply(gray)
        
        return enhanced
    
    def auto_color_correction(self, image: np.ndarray) -> np.ndarray:
        """
        Auto white balance untuk koreksi warna
        
        Args:
            image: Input image (BGR)
            
        Returns:
            numpy.ndarray: Color corrected image
        """
        if len(image.shape) == 2:
            return image
        
        # Simple gray world assumption
        result = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        avg_a = np.average(result[:, :, 1])
        avg_b = np.average(result[:, :, 2])
        
        result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
        result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
        
        result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
        return result
    
    def morphological_operations(self, image: np.ndarray, 
                                operation: str = 'open',
                                kernel_size: int = 3) -> np.ndarray:
        """
        Morphological operations untuk cleanup noise
        
        Args:
            image: Binary image
            operation: 'open', 'close', 'gradient', 'tophat', 'blackhat'
            kernel_size: Size of structuring element
            
        Returns:
            numpy.ndarray: Processed image
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                          (kernel_size, kernel_size))
        
        operations = {
            'open': cv2.MORPH_OPEN,
            'close': cv2.MORPH_CLOSE,
            'gradient': cv2.MORPH_GRADIENT,
            'tophat': cv2.MORPH_TOPHAT,
            'blackhat': cv2.MORPH_BLACKHAT
        }
        
        op = operations.get(operation, cv2.MORPH_OPEN)
        result = cv2.morphologyEx(image, op, kernel)
        
        return result
    
    def document_scan_effect(self, image: np.ndarray) -> np.ndarray:
        """
        Pipeline lengkap untuk membuat efek 'scanned document'
        
        Steps:
        1. Grayscale conversion
        2. Bilateral filter
        3. CLAHE
        4. Adaptive threshold
        5. Morphological opening
        
        Args:
            image: Input image
            
        Returns:
            numpy.ndarray: Document dengan scan effect
        """
        # 1. Grayscale
        gray = self.grayscale(image)
        
        # 2. Denoise dengan bilateral filter
        denoised = self.denoise_bilateral(gray, d=9, sigma_color=75, sigma_space=75)
        
        # 3. Enhance contrast
        enhanced = self.enhance_contrast(denoised, clip_limit=2.0)
        
        # 4. Adaptive threshold
        binary = self.adaptive_threshold(enhanced, block_size=11, c=2)
        
        # 5. Morphological opening untuk remove small noise
        cleaned = self.morphological_operations(binary, operation='open', kernel_size=2)
        
        return cleaned
    
    def enhance_colors(self, image: np.ndarray, saturation: float = 1.3) -> np.ndarray:
        """
        Enhance color saturation
        
        Args:
            image: Input image (BGR)
            saturation: Saturation multiplier (>1 = more saturated)
            
        Returns:
            numpy.ndarray: Color enhanced image
        """
        if len(image.shape) == 2:
            return image
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result
