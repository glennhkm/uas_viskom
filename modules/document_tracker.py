"""
Document Tracker Module
Real-time document tracking menggunakan webcam
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Callable
from .boundary_detector import DocumentDetector


class DocumentTracker:
    """
    Class untuk real-time document tracking
    
    Features:
    - Detect dan track dokumen dari webcam
    - Optical flow untuk smooth tracking
    - Auto capture ketika dokumen stable
    - Real-time preview dengan overlay
    """
    
    def __init__(self, camera_id: int = 0):
        """
        Initialize DocumentTracker
        
        Args:
            camera_id: Camera device ID (default 0 untuk default webcam)
        """
        self.camera_id = camera_id
        self.cap = None
        self.detector = DocumentDetector()
        self.last_corners = None
        self.stable_frames = 0
        self.stability_threshold = 10  # Frames harus stable sebelum auto capture
    
    def start_camera(self) -> bool:
        """
        Start camera capture
        
        Returns:
            bool: True jika berhasil
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            return False
        
        # Set resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def read_frame(self) -> Optional[np.ndarray]:
        """
        Read frame dari camera
        
        Returns:
            numpy.ndarray: Frame atau None jika gagal
        """
        if self.cap is None or not self.cap.isOpened():
            return None
        
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        return frame
    
    def is_stable(self, current_corners: np.ndarray, 
                  threshold: float = 15.0) -> bool:
        """
        Check apakah dokumen stable (tidak bergerak)
        
        Args:
            current_corners: Current detected corners
            threshold: Maximum pixel movement untuk dianggap stable
            
        Returns:
            bool: True jika stable
        """
        if self.last_corners is None:
            self.last_corners = current_corners
            return False
        
        # Calculate movement
        movement = np.abs(current_corners - self.last_corners)
        max_movement = np.max(movement)
        
        self.last_corners = current_corners
        
        return max_movement < threshold
    
    def draw_overlay(self, frame: np.ndarray, 
                    corners: Optional[np.ndarray] = None,
                    is_stable: bool = False) -> np.ndarray:
        """
        Draw overlay pada frame untuk visual feedback
        
        Args:
            frame: Input frame
            corners: Detected corners (optional)
            is_stable: Whether document is stable
            
        Returns:
            numpy.ndarray: Frame dengan overlay
        """
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Draw guide lines (centerlines)
        cv2.line(output, (w//2, 0), (w//2, h), (200, 200, 200), 1)
        cv2.line(output, (0, h//2), (w, h//2), (200, 200, 200), 1)
        
        # Draw corners jika detected
        if corners is not None:
            # Draw polygon
            pts = corners.reshape((-1, 1, 2)).astype(np.int32)
            color = (0, 255, 0) if is_stable else (0, 255, 255)
            cv2.polylines(output, [pts], True, color, 3)
            
            # Draw corner points
            for point in corners:
                x, y = int(point[0]), int(point[1])
                cv2.circle(output, (x, y), 8, (0, 0, 255), -1)
            
            # Status text
            status = "STABLE - Ready to Capture" if is_stable else "DETECTING..."
            color = (0, 255, 0) if is_stable else (0, 255, 255)
            cv2.putText(output, status, (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        else:
            # No document detected
            cv2.putText(output, "No Document Detected", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Instructions
        cv2.putText(output, "Press 's' to capture, 'q' to quit", (20, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Stability indicator
        if corners is not None:
            bar_width = int((self.stable_frames / self.stability_threshold) * 200)
            cv2.rectangle(output, (20, 70), (20 + bar_width, 90), (0, 255, 0), -1)
            cv2.rectangle(output, (20, 70), (220, 90), (255, 255, 255), 2)
        
        return output
    
    def track_document(self, 
                      on_capture: Optional[Callable[[np.ndarray, np.ndarray], None]] = None,
                      auto_capture: bool = False,
                      display_window: str = "Document Tracker") -> bool:
        """
        Main tracking loop
        
        Args:
            on_capture: Callback function(frame, corners) ketika capture
            auto_capture: Auto capture ketika stable
            display_window: Window name untuk display
            
        Returns:
            bool: True jika berhasil
        """
        if not self.start_camera():
            print("Error: Tidak dapat membuka camera")
            return False
        
        print("Document Tracker started...")
        print("Press 's' to capture manually")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    break
                
                # Detect document
                corners = self.detector.detect_document(frame)
                
                # Check stability
                is_stable_flag = False
                if corners is not None:
                    is_stable_flag = self.is_stable(corners)
                    
                    if is_stable_flag:
                        self.stable_frames += 1
                    else:
                        self.stable_frames = 0
                    
                    # Auto capture jika enabled dan stable cukup lama
                    if auto_capture and self.stable_frames >= self.stability_threshold:
                        if on_capture:
                            on_capture(frame.copy(), corners.copy())
                        self.stable_frames = 0
                        print("Auto captured!")
                else:
                    self.stable_frames = 0
                    self.last_corners = None
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, corners, is_stable_flag)
                
                # Show frame
                cv2.imshow(display_window, display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    # Quit
                    break
                elif key == ord('s'):
                    # Manual capture
                    if corners is not None:
                        if on_capture:
                            on_capture(frame.copy(), corners.copy())
                        print("Manually captured!")
                    else:
                        print("No document detected to capture!")
        
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
        
        return True
    
    def capture_single(self, timeout: int = 30) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Capture single document dengan timeout
        
        Args:
            timeout: Timeout dalam detik
            
        Returns:
            tuple: (frame, corners) atau None
        """
        if not self.start_camera():
            return None
        
        result = None
        start_time = cv2.getTickCount()
        
        try:
            while True:
                # Check timeout
                elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
                if elapsed > timeout:
                    print("Timeout!")
                    break
                
                # Read frame
                frame = self.read_frame()
                if frame is None:
                    break
                
                # Detect document
                corners = self.detector.detect_document(frame)
                
                # Draw overlay
                display_frame = self.draw_overlay(frame, corners, False)
                
                # Show
                cv2.imshow("Capture Document", display_frame)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('s') and corners is not None:
                    result = (frame.copy(), corners.copy())
                    break
                elif key == ord('q'):
                    break
        
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
        
        return result
    
    def get_camera_info(self) -> dict:
        """
        Get camera information
        
        Returns:
            dict: Camera properties
        """
        if self.cap is None:
            self.start_camera()
        
        if self.cap is None:
            return {}
        
        info = {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
            'backend': self.cap.getBackendName()
        }
        
        return info
