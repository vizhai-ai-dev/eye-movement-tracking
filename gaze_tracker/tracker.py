import time
import numpy as np
import cv2
from collections import deque

class GazeTracker:
    def __init__(self, suspicious_threshold=0.3, time_window=5):
        """
        Initialize the gaze tracker
        
        Args:
            suspicious_threshold: Ratio of off-screen time to be considered suspicious
            time_window: Time window (in seconds) to track gaze
        """
        # Reduce time window from 30 to 5 seconds for faster response
        self.suspicious_threshold = suspicious_threshold
        self.time_window = time_window
        
        # Buffer to store gaze data
        self.gaze_buffer = deque()
        self.last_update_time = time.time()
        
        # Statistics
        self.off_screen_ratio = 0.0
        self.status = "ok"
        
        # Consecutive off-screen frames for immediate detection
        self.consecutive_off_screen = 0
        self.consecutive_threshold = 5  # Number of consecutive frames to trigger immediate suspicious
        
        # Heatmap data
        self.heatmap_size = (100, 100)  # Size of the heatmap
        self.heatmap = np.zeros(self.heatmap_size, dtype=np.float32)
        
        # Direction weights - give higher weight to certain directions
        self.direction_weights = {
            "center": 0.0,    # On-screen
            "up": 1.0,        # Off-screen
            "down": 0.2,      # Mostly on-screen (reading/writing)
            "left": 1.0,      # Off-screen
            "right": 1.0,     # Off-screen
            "up-left": 1.0,   # Off-screen
            "up-right": 1.0,  # Off-screen
            "down-left": 0.5, # Partially off-screen
            "down-right": 0.5,# Partially off-screen
            "unknown": 0.5    # Partially off-screen
        }
    
    def update(self, gaze_direction, is_looking_at_screen, ratios):
        """
        Update the gaze tracker with new gaze data
        
        Args:
            gaze_direction: Current gaze direction
            is_looking_at_screen: Boolean indicating if the person is looking at the screen
            ratios: Dictionary with horizontal and vertical ratios
            
        Returns:
            off_screen_ratio: Ratio of time spent looking off-screen
            status: 'ok' or 'suspicious'
            heatmap: Heatmap of gaze positions
        """
        current_time = time.time()
        
        # Track consecutive off-screen frames for immediate detection
        if not is_looking_at_screen:
            self.consecutive_off_screen += 1
        else:
            self.consecutive_off_screen = 0
        
        # Add new gaze data to buffer with weighted importance
        weight = self.direction_weights.get(gaze_direction, 1.0)
        
        self.gaze_buffer.append({
            'timestamp': current_time,
            'direction': gaze_direction,
            'on_screen': is_looking_at_screen,
            'weight': weight,
            'ratios': ratios
        })
        
        # Remove old data from buffer (older than time_window)
        cutoff_time = current_time - self.time_window
        while self.gaze_buffer and self.gaze_buffer[0]['timestamp'] < cutoff_time:
            self.gaze_buffer.popleft()
        
        # Calculate weighted off-screen ratio
        if self.gaze_buffer:
            # Use weighted calculation for off-screen time
            total_weight = sum(data['weight'] for data in self.gaze_buffer if not data['on_screen'])
            max_weight = sum(data.get('weight', 1.0) for data in self.gaze_buffer)
            
            if max_weight > 0:
                self.off_screen_ratio = total_weight / len(self.gaze_buffer)
            else:
                self.off_screen_ratio = 0.0
        else:
            self.off_screen_ratio = 0.0
        
        # Update status - consider both the ratio and consecutive frames
        if self.consecutive_off_screen >= self.consecutive_threshold or self.off_screen_ratio > self.suspicious_threshold:
            self.status = "suspicious"
        else:
            self.status = "ok"
        
        # Update heatmap
        self._update_heatmap(ratios)
        
        return self.off_screen_ratio, self.status, self.get_heatmap()
    
    def _update_heatmap(self, ratios):
        """Update the gaze heatmap with new ratio data"""
        if not ratios:
            return
        
        # Map ratios to heatmap coordinates
        x = int(ratios['horizontal'] * (self.heatmap_size[1] - 1))
        y = int(ratios['vertical'] * (self.heatmap_size[0] - 1))
        
        # Ensure coordinates are within bounds
        x = max(0, min(x, self.heatmap_size[1] - 1))
        y = max(0, min(y, self.heatmap_size[0] - 1))
        
        # Update heatmap (add heat at the gaze position)
        self.heatmap[y, x] += 0.5
        
        # Apply decay to the entire heatmap (cool down over time)
        self.heatmap *= 0.99
    
    def get_heatmap(self, colored=True):
        """
        Get the gaze heatmap
        
        Args:
            colored: Boolean indicating if the heatmap should be colored
            
        Returns:
            heatmap: Normalized and optionally colored heatmap
        """
        # Normalize heatmap
        if np.max(self.heatmap) > 0:
            normalized = self.heatmap / np.max(self.heatmap)
        else:
            normalized = self.heatmap
        
        # Convert to 8-bit
        heatmap_8bit = (normalized * 255).astype(np.uint8)
        
        if colored:
            # Apply colormap
            colored_heatmap = cv2.applyColorMap(heatmap_8bit, cv2.COLORMAP_JET)
            return colored_heatmap
        else:
            return heatmap_8bit
    
    def draw_status_visualization(self, frame, gaze_direction):
        """
        Draw status visualization on the frame
        
        Args:
            frame: BGR image
            gaze_direction: Current gaze direction
            
        Returns:
            frame: Frame with status visualization
        """
        # Draw off-screen ratio bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = frame.shape[0] - 40
        
        # Draw background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        
        # Draw filled portion
        filled_width = int(bar_width * self.off_screen_ratio)
        
        # Color changes based on ratio (green to yellow to red)
        if self.off_screen_ratio < 0.2:
            color = (0, 255, 0)  # Green
        elif self.off_screen_ratio < self.suspicious_threshold:
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 0, 255)  # Red
            
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                     color, -1)
        
        # Add labels
        cv2.putText(frame, f"Off-screen: {self.off_screen_ratio:.2f}", 
                   (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        status_x = frame.shape[1] - 150
        status_y = 30
        
        status_color = (0, 255, 0) if self.status == "ok" else (0, 0, 255)
        status_text = "✓ OK" if self.status == "ok" else "⚠ SUSPICIOUS"
        
        # Add consecutive counter (for debugging)
        if gaze_direction != "center":
            cv2.putText(frame, f"Away for: {self.consecutive_off_screen} frames", 
                      (status_x - 100, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, status_text, (status_x, status_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        return frame
    
    def draw_heatmap_overlay(self, frame):
        """
        Draw the gaze heatmap overlay on the frame
        
        Args:
            frame: BGR image
            
        Returns:
            frame: Frame with heatmap overlay
        """
        # Get colored heatmap
        heatmap = self.get_heatmap(colored=True)
        
        # Resize heatmap to frame size
        resized_heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
        
        # Create a mask based on heatmap intensity
        gray_heatmap = cv2.cvtColor(resized_heatmap, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray_heatmap, 1, 255, cv2.THRESH_BINARY)
        
        # Create mask with 3 channels
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        
        # Apply mask to heatmap
        masked_heatmap = cv2.bitwise_and(resized_heatmap, mask_3ch)
        
        # Overlay heatmap on frame with transparency
        alpha = 0.3  # Transparency factor
        
        # Add the weighted heatmap to the original frame
        blended = cv2.addWeighted(frame, 1.0, masked_heatmap, alpha, 0)
        
        return blended 