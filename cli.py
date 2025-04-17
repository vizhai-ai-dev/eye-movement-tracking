import cv2
import time
import sys
from gaze_tracker.detector import FaceDetector
from gaze_tracker.gaze_estimator import GazeEstimator
from gaze_tracker.tracker import GazeTracker

def run_calibration(cap, face_detector, gaze_estimator):
    """Run the calibration process"""
    print("\nCalibration Mode")
    print("Please look at the center of the screen for 5 seconds...")
    
    calibration_samples = 0
    calibration_start_time = time.time()
    
    gaze_estimator.reset_calibration()
    
    while time.time() - calibration_start_time < 5:  # Run for 5 seconds
        # Capture a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam")
            return False
        
        # Flip the frame horizontally (mirror view)
        frame = cv2.flip(frame, 1)
        
        # Detect face landmarks
        landmarks, _ = face_detector.detect_face_landmarks(frame)
        
        # Get eye landmarks
        left_eye, right_eye, left_iris, right_iris = face_detector.get_eye_landmarks(landmarks)
        
        # Add calibration sample
        if left_eye and right_eye and left_iris and right_iris:
            sample_added = gaze_estimator.add_calibration_sample(left_eye, right_eye, left_iris, right_iris)
            if sample_added:
                calibration_samples += 1
        
        # Draw calibration instructions
        cv2.putText(frame, "CALIBRATION: Please look at the center of the screen", 
                   (frame.shape[1]//2 - 200, frame.shape[0]//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw a target in the center
        center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 255), 2)
        cv2.line(frame, (center_x - 30, center_y), (center_x + 30, center_y), (0, 255, 255), 2)
        cv2.line(frame, (center_x, center_y - 30), (center_x, center_y + 30), (0, 255, 255), 2)
        
        # Draw progress
        progress = min(100, int((time.time() - calibration_start_time) / 5 * 100))
        cv2.rectangle(frame, (center_x - 50, center_y + 50), (center_x + 50, center_y + 70), (255, 255, 255), 2)
        cv2.rectangle(frame, (center_x - 50, center_y + 50), (center_x - 50 + int(progress), center_y + 70), (0, 255, 0), -1)
        
        # Display the frame
        cv2.imshow('Eye Movement Tracking - Calibration', frame)
        
        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
        
        # Update progress in console
        sys.stdout.write(f"\rCalibration progress: {progress}% - Samples: {calibration_samples}")
        sys.stdout.flush()
        
        # Add a small delay
        time.sleep(0.02)  # Higher frame rate (was 0.03)
    
    # Finalize calibration if not already done
    if not gaze_estimator.is_calibrated:
        gaze_estimator._finalize_calibration()
    
    print("\nCalibration complete!")
    return True

def main():
    print("Starting Eye Movement Tracking System (CLI Mode)")
    print("Press 'q' to exit, 'c' to calibrate, '+/-' to adjust sensitivity")
    
    # Initialize the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS for smoother processing
    
    # Initialize face detector, gaze estimator, and gaze tracker
    face_detector = FaceDetector(min_detection_confidence=0.5)
    gaze_estimator = GazeEstimator()
    gaze_tracker = GazeTracker(suspicious_threshold=0.3, time_window=5)  # Reduced time window
    
    # Run initial calibration
    run_calibration(cap, face_detector, gaze_estimator)
    
    # Start processing frames
    try:
        consecutive_frames = 0
        last_direction = "center"
        
        while True:
            # Capture a frame from the webcam
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame from webcam")
                break
            
            # Flip the frame horizontally (mirror view)
            frame = cv2.flip(frame, 1)
            
            # Detect face landmarks
            landmarks, _ = face_detector.detect_face_landmarks(frame)
            
            # Get eye landmarks
            left_eye, right_eye, left_iris, right_iris = face_detector.get_eye_landmarks(landmarks)
            
            # Draw eye landmarks
            frame = face_detector.draw_eye_landmarks(frame, left_eye, right_eye, left_iris, right_iris)
            
            # Estimate gaze direction
            gaze_direction, is_looking_at_screen, ratios = gaze_estimator.estimate_gaze_direction(
                left_eye, right_eye, left_iris, right_iris
            )
            
            # Count consecutive off-center frames
            if gaze_direction != "center" and gaze_direction == last_direction:
                consecutive_frames += 1
            else:
                consecutive_frames = 0
            
            last_direction = gaze_direction
            
            # Draw gaze visualization
            frame = gaze_estimator.draw_gaze_visualization(
                frame, left_eye, right_eye, left_iris, right_iris, gaze_direction
            )
            
            # Update gaze tracker
            off_screen_ratio, status, _ = gaze_tracker.update(
                gaze_direction, is_looking_at_screen, ratios
            )
            
            # Draw status visualization
            frame = gaze_tracker.draw_status_visualization(frame, gaze_direction)
            
            # Optional: Draw heatmap overlay
            frame = gaze_tracker.draw_heatmap_overlay(frame)
            
            # Add offset visualization - draw an indicator showing how far from center
            if 'h_offset' in ratios and 'v_offset' in ratios:
                h_offset = ratios['h_offset']
                v_offset = ratios['v_offset']
                
                # Draw crosshair in center of frame
                center_x, center_y = frame.shape[1]//2, frame.shape[0]//2
                
                # Draw indicator dot - scaled position shows gaze offset
                indicator_scale = 100  # Scale factor for visualization
                indicator_x = center_x + int(h_offset * indicator_scale)
                indicator_y = center_y + int(v_offset * indicator_scale)
                
                # Keep indicator within frame bounds
                indicator_x = max(0, min(frame.shape[1]-1, indicator_x))
                indicator_y = max(0, min(frame.shape[0]-1, indicator_y))
                
                # Draw indicator
                cv2.circle(frame, (indicator_x, indicator_y), 8, (0, 165, 255), -1)
                cv2.circle(frame, (center_x, center_y), 2, (255, 255, 255), -1)
                
                # Connect center to indicator
                cv2.line(frame, (center_x, center_y), (indicator_x, indicator_y), (0, 165, 255), 2)
            
            # Add settings information
            settings_text = f"Threshold: {gaze_tracker.suspicious_threshold:.2f} | Window: {gaze_tracker.time_window}s"
            cv2.putText(frame, settings_text, (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Eye Movement Tracking', frame)
            
            # Print status information to console
            sys.stdout.write(f"\rGaze: {gaze_direction.ljust(10)} | Off-screen: {off_screen_ratio:.2f} | Status: {status}")
            sys.stdout.flush()
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                run_calibration(cap, face_detector, gaze_estimator)
            elif key == ord('+') or key == ord('='):  # More sensitive
                gaze_tracker.suspicious_threshold = max(0.1, gaze_tracker.suspicious_threshold - 0.05)
                print(f"\nIncreased sensitivity: threshold = {gaze_tracker.suspicious_threshold:.2f}")
            elif key == ord('-') or key == ord('_'):  # Less sensitive
                gaze_tracker.suspicious_threshold = min(0.9, gaze_tracker.suspicious_threshold + 0.05)
                print(f"\nDecreased sensitivity: threshold = {gaze_tracker.suspicious_threshold:.2f}")
                
            # Add a small delay to control the frame rate
            time.sleep(0.02)  # ~50 FPS (reduced from 0.03/30 FPS)
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        print("\nResources released. Exiting.")

if __name__ == "__main__":
    main() 