"""
Utility functions untuk Smart Document Scanner
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import matplotlib.pyplot as plt


def load_image(image_path: str) -> np.ndarray:
    """
    Load image dari file path
    
    Args:
        image_path: Path ke file gambar
        
    Returns:
        numpy.ndarray: Image dalam format BGR
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Tidak dapat membaca gambar: {image_path}")
    return image


def save_image(image: np.ndarray, output_path: str) -> bool:
    """
    Simpan image ke file
    
    Args:
        image: Image dalam format BGR/Grayscale
        output_path: Path output file
        
    Returns:
        bool: True jika berhasil
    """
    return cv2.imwrite(output_path, image)


def resize_image(image: np.ndarray, width: Optional[int] = None, 
                 height: Optional[int] = None, 
                 max_dimension: Optional[int] = None) -> np.ndarray:
    """
    Resize image dengan mempertahankan aspect ratio
    
    Args:
        image: Input image
        width: Target width (optional)
        height: Target height (optional)
        max_dimension: Maximum dimension untuk width/height (optional)
        
    Returns:
        numpy.ndarray: Resized image
    """
    h, w = image.shape[:2]
    
    if max_dimension:
        if max(h, w) > max_dimension:
            if w > h:
                width = max_dimension
            else:
                height = max_dimension
    
    if width and not height:
        ratio = width / w
        height = int(h * ratio)
    elif height and not width:
        ratio = height / h
        width = int(w * ratio)
    elif not width and not height:
        return image
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Order points dalam urutan: top-left, top-right, bottom-right, bottom-left
    
    Args:
        pts: Array of 4 points
        
    Returns:
        numpy.ndarray: Ordered points
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left point akan memiliki sum terkecil
    # Bottom-right point akan memiliki sum terbesar
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right point akan memiliki diff terkecil
    # Bottom-left point akan memiliki diff terbesar
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def display_images(images: List[np.ndarray], titles: List[str], 
                   figsize: Tuple[int, int] = (15, 5), cmap: str = 'gray'):
    """
    Display multiple images dalam satu figure
    
    Args:
        images: List of images
        titles: List of titles
        figsize: Figure size
        cmap: Colormap untuk grayscale images
    """
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    
    if n == 1:
        axes = [axes]
    
    for i, (img, title) in enumerate(zip(images, titles)):
        if len(img.shape) == 2:  # Grayscale
            axes[i].imshow(img, cmap=cmap)
        else:  # Color - convert BGR to RGB
            axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(title)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def draw_contour(image: np.ndarray, contour: np.ndarray, 
                 color: Tuple[int, int, int] = (0, 255, 0), 
                 thickness: int = 3) -> np.ndarray:
    """
    Draw contour pada image
    
    Args:
        image: Input image
        contour: Contour points
        color: Line color (BGR)
        thickness: Line thickness
        
    Returns:
        numpy.ndarray: Image dengan contour
    """
    output = image.copy()
    cv2.drawContours(output, [contour], -1, color, thickness)
    return output


def calculate_aspect_ratio(width: int, height: int) -> float:
    """
    Calculate aspect ratio
    
    Args:
        width: Image width
        height: Image height
        
    Returns:
        float: Aspect ratio
    """
    return width / height


def validate_image_format(image: np.ndarray) -> bool:
    """
    Validasi format image
    
    Args:
        image: Input image
        
    Returns:
        bool: True jika valid
    """
    if image is None or not isinstance(image, np.ndarray):
        return False
    if len(image.shape) not in [2, 3]:
        return False
    return True


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert image ke grayscale
    
    Args:
        image: Input image (BGR or already grayscale)
        
    Returns:
        numpy.ndarray: Grayscale image
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def add_padding(image: np.ndarray, padding: int, 
                color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Add padding ke image
    
    Args:
        image: Input image
        padding: Padding size
        color: Padding color
        
    Returns:
        numpy.ndarray: Image dengan padding
    """
    if len(image.shape) == 2:
        return cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=color[0])
    else:
        return cv2.copyMakeBorder(image, padding, padding, padding, padding,
                                 cv2.BORDER_CONSTANT, value=color)


def calculate_distance(pt1: Tuple[int, int], pt2: Tuple[int, int]) -> float:
    """
    Calculate Euclidean distance antara dua points
    
    Args:
        pt1: Point 1 (x, y)
        pt2: Point 2 (x, y)
        
    Returns:
        float: Distance
    """
    return np.sqrt((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)
