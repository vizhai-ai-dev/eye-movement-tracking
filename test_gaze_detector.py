"""
Simple test script to verify that the gaze detection system is working correctly.
This script will:
1. Capture a single frame from the webcam
2. Run face detection and eye landmark detection
3. Estimate gaze direction
4. Display the results
"""

import cv2
import time
import numpy as np
from gaze_tracker.detector import FaceDetector
from gaze_tracker.gaze_estimator import GazeEstimator

def test_face_detection():
    """Test the face detection functionality"""
    print("\n=== Testing Face Detection ===")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    print("Webcam opened successfully")
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize face detector
    print("Initializing face detector...")
    face_detector = FaceDetector(min_detection_confidence=0.5)
    
    # Capture a frame
    print("Capturing frame...")
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        cap.release()
        return False
    
    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    
    # Detect face landmarks
    print("Detecting face landmarks...")
    start_time = time.time()
    landmarks, _ = face_detector.detect_face_landmarks(frame)
    detection_time = time.time() - start_time
    
    if landmarks:
        print(f"✅ Face detected in {detection_time:.2f} seconds")
        print(f"   Number of landmarks: {len(landmarks)}")
    else:
        print("❌ No face detected")
        cap.release()
        return False
    
    # Get eye landmarks
    print("\nExtracting eye landmarks...")
    left_eye, right_eye, left_iris, right_iris = face_detector.get_eye_landmarks(landmarks)
    
    if left_eye and right_eye:
        print(f"✅ Eye landmarks detected")
        print(f"   Left eye landmarks: {len(left_eye)}")
        print(f"   Right eye landmarks: {len(right_eye)}")
        print(f"   Left iris landmarks: {len(left_iris) if left_iris else 0}")
        print(f"   Right iris landmarks: {len(right_iris) if right_iris else 0}")
    else:
        print("❌ Eye landmarks not detected")
        cap.release()
        return False
    
    # Draw eye landmarks on the frame for visualization
    marked_frame = frame.copy()
    marked_frame = face_detector.draw_eye_landmarks(marked_frame, left_eye, right_eye, left_iris, right_iris)
    
    # Save the frame with landmarks for review
    print("\nSaving test image with landmarks...")
    cv2.imwrite("test_landmarks.jpg", marked_frame)
    print("✅ Test image saved as 'test_landmarks.jpg'")
    
    # Release the webcam
    cap.release()
    
    return True

def test_gaze_estimation():
    """Test the gaze estimation functionality"""
    print("\n=== Testing Gaze Estimation ===")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Initialize face detector and gaze estimator
    print("Initializing face detector and gaze estimator...")
    face_detector = FaceDetector(min_detection_confidence=0.5)
    gaze_estimator = GazeEstimator()
    
    # Capture a frame
    print("Capturing frame...")
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        cap.release()
        return False
    
    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    
    # Detect face landmarks
    print("Detecting face landmarks...")
    landmarks, _ = face_detector.detect_face_landmarks(frame)
    
    if not landmarks:
        print("❌ No face detected")
        cap.release()
        return False
    
    # Get eye landmarks
    print("Extracting eye landmarks...")
    left_eye, right_eye, left_iris, right_iris = face_detector.get_eye_landmarks(landmarks)
    
    if not (left_eye and right_eye):
        print("❌ Eye landmarks not detected")
        cap.release()
        return False
    
    # Estimate gaze direction
    print("Estimating gaze direction...")
    start_time = time.time()
    gaze_direction, is_looking_at_screen, ratios = gaze_estimator.estimate_gaze_direction(
        left_eye, right_eye, left_iris, right_iris
    )
    estimation_time = time.time() - start_time
    
    print(f"✅ Gaze direction estimated in {estimation_time:.2f} seconds")
    print(f"   Gaze direction: {gaze_direction}")
    print(f"   Looking at screen: {'Yes' if is_looking_at_screen else 'No'}")
    print(f"   Horizontal ratio: {ratios['horizontal']:.2f}")
    print(f"   Vertical ratio: {ratios['vertical']:.2f}")
    
    # Draw gaze visualization on the frame
    visualized_frame = frame.copy()
    visualized_frame = gaze_estimator.draw_gaze_visualization(
        visualized_frame, left_eye, right_eye, left_iris, right_iris, gaze_direction
    )
    
    # Save the frame with gaze visualization for review
    print("\nSaving test image with gaze visualization...")
    cv2.imwrite("test_gaze.jpg", visualized_frame)
    print("✅ Test image saved as 'test_gaze.jpg'")
    
    # Release the webcam
    cap.release()
    
    return True

def main():
    """Run all tests"""
    print("==================================")
    print("  Eye Movement Tracking System Tests")
    print("==================================\n")
    
    # Test face detection
    face_detection_result = test_face_detection()
    
    # Test gaze estimation
    gaze_estimation_result = test_gaze_estimation()
    
    # Print summary
    print("\n==================================")
    print("  Test Summary")
    print("==================================")
    print(f"Face Detection: {'✅ PASSED' if face_detection_result else '❌ FAILED'}")
    print(f"Gaze Estimation: {'✅ PASSED' if gaze_estimation_result else '❌ FAILED'}")
    
    if face_detection_result and gaze_estimation_result:
        print("\n🎉 All tests passed! The system is working correctly.")
        print("You can now run the full application:")
        print("  - For CLI version: python cli.py")
        print("  - For web version: python app.py (access at http://localhost:8080)")
    else:
        print("\n❌ Some tests failed. Please check the issues before running the full application.")

if __name__ == "__main__":
    main() 