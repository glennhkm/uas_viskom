"""
Seam Carving Module
Content-aware image resizing dan object removal
"""

import cv2
import numpy as np
from typing import Tuple, Optional


class SeamCarver:
    """
    Class untuk seam carving - content-aware image resizing
    
    Seam Carving Algorithm:
    1. Calculate energy map (gradient magnitude)
    2. Find minimum energy seam menggunakan dynamic programming
    3. Remove/insert seam
    4. Repeat sampai target size
    
    Aplikasi:
    - Remove watermark
    - Remove objek tidak diinginkan
    - Resize image dengan minimal distortion
    """
    
    def __init__(self):
        """Initialize SeamCarver"""
        pass
    
    def calculate_energy(self, image: np.ndarray) -> np.ndarray:
        """
        Calculate energy map dari image
        
        Energy = gradient magnitude
        Tinggi gradient = high energy (edges, important features)
        Rendah gradient = low energy (uniform areas)
        
        Args:
            image: Input image (grayscale atau color)
            
        Returns:
            numpy.ndarray: Energy map
        """
        # Convert ke grayscale jika perlu
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate gradients menggunakan Sobel operator
        # Sobel X - horizontal edges
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        # Sobel Y - vertical edges
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Energy = magnitude of gradient
        energy = np.sqrt(sobel_x**2 + sobel_y**2)
        
        return energy
    
    def find_vertical_seam(self, energy: np.ndarray) -> np.ndarray:
        """
        Find seam vertikal dengan minimum energy
        
        Dynamic Programming approach:
        - M[i, j] = minimum cumulative energy dari top ke (i, j)
        - M[i, j] = energy[i, j] + min(M[i-1, j-1], M[i-1, j], M[i-1, j+1])
        
        Args:
            energy: Energy map
            
        Returns:
            numpy.ndarray: Array of column indices untuk seam
        """
        rows, cols = energy.shape
        
        # Initialize cumulative energy map
        M = energy.copy()
        
        # Backtrack array untuk reconstruction
        backtrack = np.zeros_like(M, dtype=int)
        
        # Fill cumulative energy map (dynamic programming)
        for i in range(1, rows):
            for j in range(cols):
                # Three possible parents: up-left, up, up-right
                if j == 0:
                    # Left edge
                    idx = np.argmin(M[i-1, j:j+2])
                    backtrack[i, j] = idx + j
                    M[i, j] += M[i-1, idx + j]
                elif j == cols - 1:
                    # Right edge
                    idx = np.argmin(M[i-1, j-1:j+1])
                    backtrack[i, j] = idx + j - 1
                    M[i, j] += M[i-1, idx + j - 1]
                else:
                    # Middle
                    idx = np.argmin(M[i-1, j-1:j+2])
                    backtrack[i, j] = idx + j - 1
                    M[i, j] += M[i-1, idx + j - 1]
        
        # Backtrack untuk find seam
        seam = np.zeros(rows, dtype=int)
        
        # Start dari bottom row - minimum energy
        seam[-1] = np.argmin(M[-1])
        
        # Backtrack ke atas
        for i in range(rows - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]
        
        return seam
    
    def remove_vertical_seam(self, image: np.ndarray, seam: np.ndarray) -> np.ndarray:
        """
        Remove vertical seam dari image
        
        Args:
            image: Input image
            seam: Array of column indices
            
        Returns:
            numpy.ndarray: Image dengan seam removed (width - 1)
        """
        rows, cols = image.shape[:2]
        
        if len(image.shape) == 3:
            # Color image
            output = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
            for i in range(rows):
                col = seam[i]
                output[i, :, 0] = np.delete(image[i, :, 0], col)
                output[i, :, 1] = np.delete(image[i, :, 1], col)
                output[i, :, 2] = np.delete(image[i, :, 2], col)
        else:
            # Grayscale image
            output = np.zeros((rows, cols - 1), dtype=image.dtype)
            for i in range(rows):
                col = seam[i]
                output[i, :] = np.delete(image[i, :], col)
        
        return output
    
    def insert_vertical_seam(self, image: np.ndarray, seam: np.ndarray) -> np.ndarray:
        """
        Insert vertical seam ke image (untuk enlarging)
        
        Args:
            image: Input image
            seam: Array of column indices
            
        Returns:
            numpy.ndarray: Image dengan seam inserted (width + 1)
        """
        rows, cols = image.shape[:2]
        
        if len(image.shape) == 3:
            # Color image
            output = np.zeros((rows, cols + 1, 3), dtype=image.dtype)
            for i in range(rows):
                col = seam[i]
                for ch in range(3):
                    if col == 0:
                        pixel_value = image[i, col, ch]
                    else:
                        # Average of left and right pixels
                        pixel_value = (int(image[i, col, ch]) + int(image[i, col - 1, ch])) // 2
                    
                    output[i, :, ch] = np.insert(image[i, :, ch], col, pixel_value)
        else:
            # Grayscale
            output = np.zeros((rows, cols + 1), dtype=image.dtype)
            for i in range(rows):
                col = seam[i]
                if col == 0:
                    pixel_value = image[i, col]
                else:
                    pixel_value = (int(image[i, col]) + int(image[i, col - 1])) // 2
                
                output[i, :] = np.insert(image[i, :], col, pixel_value)
        
        return output
    
    def resize_width(self, image: np.ndarray, target_width: int) -> np.ndarray:
        """
        Resize image ke target width menggunakan seam carving
        
        Args:
            image: Input image
            target_width: Target width
            
        Returns:
            numpy.ndarray: Resized image
        """
        current_width = image.shape[1]
        output = image.copy()
        
        if target_width < current_width:
            # Shrink - remove seams
            num_seams = current_width - target_width
            for _ in range(num_seams):
                energy = self.calculate_energy(output)
                seam = self.find_vertical_seam(energy)
                output = self.remove_vertical_seam(output, seam)
        elif target_width > current_width:
            # Enlarge - insert seams
            num_seams = target_width - current_width
            seams_to_insert = []
            
            # Find all seams first
            temp = output.copy()
            for _ in range(num_seams):
                energy = self.calculate_energy(temp)
                seam = self.find_vertical_seam(energy)
                seams_to_insert.append(seam.copy())
                temp = self.remove_vertical_seam(temp, seam)
            
            # Insert seams (in reverse order)
            for seam in reversed(seams_to_insert):
                output = self.insert_vertical_seam(output, seam)
        
        return output
    
    def remove_object(self, image: np.ndarray, mask: np.ndarray,
                     max_iterations: int = None) -> np.ndarray:
        """
        Remove object dari image berdasarkan mask
        
        Process:
        1. Set energy di area mask sangat rendah (agar diprioritaskan untuk removal)
        2. Remove seams yang melewati area mask
        3. Repeat sampai object hilang
        
        Args:
            image: Input image
            mask: Binary mask (255 = object to remove, 0 = keep)
            max_iterations: Maximum number of seams to remove
            
        Returns:
            numpy.ndarray: Image dengan object removed
        """
        output = image.copy()
        mask_copy = mask.copy()
        
        # Calculate max iterations jika tidak di-specify
        if max_iterations is None:
            # Count pixels dalam mask
            max_iterations = np.sum(mask > 0) // image.shape[0]
            max_iterations = min(max_iterations, image.shape[1] // 2)
        
        iterations = 0
        while np.any(mask_copy > 0) and iterations < max_iterations:
            # Calculate energy
            energy = self.calculate_energy(output)
            
            # Set energy di area mask sangat rendah (high priority untuk removal)
            energy[mask_copy > 0] = -1000
            
            # Find dan remove seam
            seam = self.find_vertical_seam(energy)
            output = self.remove_vertical_seam(output, seam)
            
            # Update mask
            mask_copy = self.remove_vertical_seam(mask_copy, seam)
            
            iterations += 1
        
        return output
    
    def visualize_seam(self, image: np.ndarray, seam: np.ndarray,
                      color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Visualize seam pada image (untuk debugging)
        
        Args:
            image: Input image
            seam: Seam array
            color: Seam color (BGR)
            
        Returns:
            numpy.ndarray: Image dengan seam highlighted
        """
        output = image.copy()
        
        for i, col in enumerate(seam):
            if len(output.shape) == 3:
                output[i, col] = color
            else:
                output[i, col] = 255
        
        return output
