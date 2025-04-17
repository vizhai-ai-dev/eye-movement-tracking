import cv2
import numpy as np
import math
import time

class GazeEstimator:
    def __init__(self):
        # Default threshold values for gaze direction
        self.horizontal_ratio_threshold = 0.25  # Threshold for left/right gaze (reduced from 0.42)
        self.vertical_ratio_threshold = 0.30    # Threshold for up/down gaze (reduced from 0.55)
        
        # Buffer for gaze directions (to reduce noise)
        self.direction_buffer = []
        self.buffer_size = 3  # Reduced from 5 for quicker response
        
        # Calibration data
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_needed = True
        self.center_horizontal_ratio = 0.5  # Default center position (will be calibrated)
        self.center_vertical_ratio = 0.5    # Default center position (will be calibrated)
        
    def _calculate_eye_center(self, eye_landmarks):
        """Calculate the center of the eye from eye landmarks"""
        if not eye_landmarks:
            return None
        
        # Calculate the center of the eye
        center_x = sum(p[0] for p in eye_landmarks) / len(eye_landmarks)
        center_y = sum(p[1] for p in eye_landmarks) / len(eye_landmarks)
        
        return (int(center_x), int(center_y))
        
    def _calculate_iris_center(self, iris_landmarks):
        """Calculate the center of the iris from iris landmarks"""
        if not iris_landmarks:
            return None
        
        # Calculate the center of the iris
        center_x = sum(p[0] for p in iris_landmarks) / len(iris_landmarks)
        center_y = sum(p[1] for p in iris_landmarks) / len(iris_landmarks)
        
        return (int(center_x), int(center_y))
    
    def _get_eye_box(self, eye_landmarks):
        """Get the bounding box of the eye"""
        if not eye_landmarks:
            return None
        
        x_coords = [p[0] for p in eye_landmarks]
        y_coords = [p[1] for p in eye_landmarks]
        
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        eye_width = max_x - min_x
        eye_height = max_y - min_y
        
        return (min_x, min_y, eye_width, eye_height)
    
    def _calculate_gaze_ratio(self, eye_box, iris_center):
        """Calculate the gaze ratio (how far the iris is from the center of the eye)"""
        if not eye_box or not iris_center:
            return 0.5, 0.5  # Default to center
        
        min_x, min_y, eye_width, eye_height = eye_box
        iris_x, iris_y = iris_center
        
        # Calculate horizontal and vertical ratio
        # 0.0 means looking far left/up, 1.0 means looking far right/down
        horizontal_ratio = (iris_x - min_x) / eye_width if eye_width > 0 else 0.5
        vertical_ratio = (iris_y - min_y) / eye_height if eye_height > 0 else 0.5
        
        return horizontal_ratio, vertical_ratio
    
    def _get_gaze_direction(self, horizontal_ratio, vertical_ratio):
        """Determine gaze direction based on horizontal and vertical ratios"""
        # Calculate offsets from calibrated center positions
        h_offset = horizontal_ratio - self.center_horizontal_ratio
        v_offset = vertical_ratio - self.center_vertical_ratio
        
        # Horizontal direction
        if h_offset < -self.horizontal_ratio_threshold:
            h_direction = "left"
        elif h_offset > self.horizontal_ratio_threshold:
            h_direction = "right"
        else:
            h_direction = "center"
        
        # Vertical direction
        if v_offset < -self.vertical_ratio_threshold:
            v_direction = "up"
        elif v_offset > self.vertical_ratio_threshold:
            v_direction = "down"
        else:
            v_direction = "center"
        
        # Combine directions
        if h_direction == "center" and v_direction == "center":
            return "center"
        elif h_direction != "center" and v_direction == "center":
            return h_direction
        elif h_direction == "center" and v_direction != "center":
            return v_direction
        else:
            return f"{v_direction}-{h_direction}"
    
    def _smooth_direction(self, direction):
        """Apply smoothing to reduce flickering of gaze direction"""
        self.direction_buffer.append(direction)
        if len(self.direction_buffer) > self.buffer_size:
            self.direction_buffer.pop(0)
        
        # Count occurrences of each direction
        direction_counts = {}
        for d in self.direction_buffer:
            direction_counts[d] = direction_counts.get(d, 0) + 1
        
        # Return the most common direction
        return max(direction_counts, key=direction_counts.get)
    
    def add_calibration_sample(self, left_eye, right_eye, left_iris, right_iris):
        """
        Add a calibration sample when user is looking at center of screen
        
        Args:
            left_eye, right_eye, left_iris, right_iris: Eye landmarks
            
        Returns:
            bool: Whether calibration is complete
        """
        if not left_eye or not right_eye or not left_iris or not right_iris:
            return False
        
        # Calculate eye boxes
        left_eye_box = self._get_eye_box(left_eye)
        right_eye_box = self._get_eye_box(right_eye)
        
        # Calculate iris centers
        left_iris_center = self._calculate_iris_center(left_iris)
        right_iris_center = self._calculate_iris_center(right_iris)
        
        # Calculate gaze ratios
        left_h_ratio, left_v_ratio = self._calculate_gaze_ratio(left_eye_box, left_iris_center)
        right_h_ratio, right_v_ratio = self._calculate_gaze_ratio(right_eye_box, right_iris_center)
        
        # Average ratios from both eyes
        avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
        avg_v_ratio = (left_v_ratio + right_v_ratio) / 2
        
        # Add sample to calibration data
        self.calibration_samples.append((avg_h_ratio, avg_v_ratio))
        
        # If we have enough samples, finalize calibration
        if len(self.calibration_samples) >= 30:  # Collect 30 samples (about 1 second)
            self._finalize_calibration()
            return True
        
        return False
    
    def _finalize_calibration(self):
        """Calculate calibration values from collected samples"""
        if not self.calibration_samples:
            return
        
        # Calculate the average center position
        h_ratios = [sample[0] for sample in self.calibration_samples]
        v_ratios = [sample[1] for sample in self.calibration_samples]
        
        # Filter out outliers (samples that are more than 2 standard deviations away)
        h_mean = np.mean(h_ratios)
        h_std = np.std(h_ratios)
        v_mean = np.mean(v_ratios)
        v_std = np.std(v_ratios)
        
        filtered_h_ratios = [r for r in h_ratios if abs(r - h_mean) < 2 * h_std]
        filtered_v_ratios = [r for r in v_ratios if abs(r - v_mean) < 2 * v_std]
        
        # Set the calibrated center positions
        self.center_horizontal_ratio = np.mean(filtered_h_ratios)
        self.center_vertical_ratio = np.mean(filtered_v_ratios)
        
        # Adjust thresholds based on observed variance - set narrower thresholds for more sensitivity
        self.horizontal_ratio_threshold = max(0.05, min(0.25, 1.5 * np.std(filtered_h_ratios)))
        self.vertical_ratio_threshold = max(0.05, min(0.25, 1.5 * np.std(filtered_v_ratios)))
        
        print(f"Calibration complete:")
        print(f"  Center position: H={self.center_horizontal_ratio:.2f}, V={self.center_vertical_ratio:.2f}")
        print(f"  Thresholds: H={self.horizontal_ratio_threshold:.2f}, V={self.vertical_ratio_threshold:.2f}")
        
        self.is_calibrated = True
        self.calibration_needed = False
        self.calibration_samples = []
    
    def reset_calibration(self):
        """Reset calibration data"""
        self.is_calibrated = False
        self.calibration_samples = []
        self.calibration_needed = True
        self.center_horizontal_ratio = 0.5
        self.center_vertical_ratio = 0.5
        self.horizontal_ratio_threshold = 0.25  # Default threshold
        self.vertical_ratio_threshold = 0.30    # Default threshold
    
    def estimate_gaze_direction(self, left_eye, right_eye, left_iris, right_iris):
        """
        Estimate the gaze direction based on eye and iris landmarks
        
        Args:
            left_eye: List of left eye landmarks
            right_eye: List of right eye landmarks
            left_iris: List of left iris landmarks
            right_iris: List of right iris landmarks
            
        Returns:
            direction: Estimated gaze direction (left, right, center, up, down)
            is_looking_at_screen: Boolean indicating if the person is looking at the screen
            ratio: Dictionary with horizontal and vertical ratios
        """
        if not left_eye or not right_eye or not left_iris or not right_iris:
            return "unknown", False, {"horizontal": 0.5, "vertical": 0.5}
        
        # Calculate eye boxes
        left_eye_box = self._get_eye_box(left_eye)
        right_eye_box = self._get_eye_box(right_eye)
        
        # Calculate iris centers
        left_iris_center = self._calculate_iris_center(left_iris)
        right_iris_center = self._calculate_iris_center(right_iris)
        
        # Calculate gaze ratios
        left_h_ratio, left_v_ratio = self._calculate_gaze_ratio(left_eye_box, left_iris_center)
        right_h_ratio, right_v_ratio = self._calculate_gaze_ratio(right_eye_box, right_iris_center)
        
        # Average ratios from both eyes
        avg_h_ratio = (left_h_ratio + right_h_ratio) / 2
        avg_v_ratio = (left_v_ratio + right_v_ratio) / 2
        
        # Get gaze direction
        direction = self._get_gaze_direction(avg_h_ratio, avg_v_ratio)
        
        # Apply smoothing (optional - can be disabled for more immediate response)
        if self.buffer_size > 1:
            smoothed_direction = self._smooth_direction(direction)
        else:
            smoothed_direction = direction
        
        # Determine if looking at screen (center gaze)
        is_looking_at_screen = smoothed_direction == "center"
        
        # Calculate offsets from center for visualization
        h_offset = avg_h_ratio - self.center_horizontal_ratio
        v_offset = avg_v_ratio - self.center_vertical_ratio
        
        return smoothed_direction, is_looking_at_screen, {
            "horizontal": avg_h_ratio, 
            "vertical": avg_v_ratio,
            "h_offset": h_offset,
            "v_offset": v_offset
        }
    
    def draw_gaze_visualization(self, frame, left_eye, right_eye, left_iris, right_iris, gaze_direction):
        """
        Draw visualization of gaze on the frame
        
        Args:
            frame: BGR image
            left_eye, right_eye, left_iris, right_iris: Eye landmarks
            gaze_direction: Estimated gaze direction
            
        Returns:
            frame: Frame with gaze visualization
        """
        if not left_eye or not right_eye:
            return frame
        
        # Draw eye centers
        left_eye_center = self._calculate_eye_center(left_eye)
        right_eye_center = self._calculate_eye_center(right_eye)
        
        # Draw iris centers
        left_iris_center = self._calculate_iris_center(left_iris)
        right_iris_center = self._calculate_iris_center(right_iris)
        
        if left_eye_center and right_eye_center:
            cv2.circle(frame, left_eye_center, 3, (0, 0, 255), -1)
            cv2.circle(frame, right_eye_center, 3, (0, 0, 255), -1)
        
        if left_iris_center and right_iris_center:
            cv2.circle(frame, left_iris_center, 3, (255, 0, 255), -1)
            cv2.circle(frame, right_iris_center, 3, (255, 0, 255), -1)
            
            # Draw line from eye center to iris center to show gaze direction
            if left_eye_center and right_eye_center:
                cv2.line(frame, left_eye_center, left_iris_center, (0, 255, 255), 1)
                cv2.line(frame, right_eye_center, right_iris_center, (0, 255, 255), 1)
        
        # Add gaze direction text
        # Color code based on direction (green for center, red for off-center)
        if gaze_direction == "center":
            text_color = (0, 255, 0)  # Green for center
        else:
            text_color = (0, 0, 255)  # Red for off-center
            
        cv2.putText(frame, f"Looking: {gaze_direction}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
        
        # Add thresholds for debugging
        cv2.putText(frame, f"Thres: H={self.horizontal_ratio_threshold:.2f}, V={self.vertical_ratio_threshold:.2f}", 
                   (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add calibration status
        if self.calibration_needed:
            cv2.putText(frame, "CALIBRATION NEEDED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif not self.is_calibrated:
            progress = len(self.calibration_samples) / 30.0 * 100
            cv2.putText(frame, f"Calibrating: {progress:.0f}%", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "Calibrated", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return frame 