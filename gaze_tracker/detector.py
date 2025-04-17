import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        # Eye landmarks indices from MediaPipe Face Mesh
        # Left eye landmarks
        self.LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        # Right eye landmarks
        self.RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        # Iris landmarks (left and right)
        self.LEFT_IRIS_INDICES = [474, 475, 476, 477]
        self.RIGHT_IRIS_INDICES = [469, 470, 471, 472]

    def detect_face_landmarks(self, frame):
        """
        Detect face landmarks in the given frame
        
        Args:
            frame: BGR image
            
        Returns:
            landmarks: Normalized face landmarks if face detected, None otherwise
            processed_frame: Frame with landmarks drawn if face detected
        """
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_height, frame_width = frame.shape[:2]
        
        # Process the frame and find face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        if not results.multi_face_landmarks:
            return None, frame
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmarks.append((x, y))
        
        return landmarks, frame

    def get_eye_landmarks(self, landmarks):
        """
        Extract eye landmarks from face landmarks
        
        Args:
            landmarks: List of face landmarks
            
        Returns:
            left_eye: List of left eye landmarks
            right_eye: List of right eye landmarks
            left_iris: List of left iris landmarks
            right_iris: List of right iris landmarks
        """
        if not landmarks:
            return None, None, None, None
        
        left_eye = [landmarks[i] for i in self.LEFT_EYE_INDICES]
        right_eye = [landmarks[i] for i in self.RIGHT_EYE_INDICES]
        left_iris = [landmarks[i] for i in self.LEFT_IRIS_INDICES]
        right_iris = [landmarks[i] for i in self.RIGHT_IRIS_INDICES]
        
        return left_eye, right_eye, left_iris, right_iris
    
    def draw_eye_landmarks(self, frame, left_eye, right_eye, left_iris, right_iris):
        """
        Draw eye landmarks on the frame
        
        Args:
            frame: BGR image
            left_eye: List of left eye landmarks
            right_eye: List of right eye landmarks
            left_iris: List of left iris landmarks
            right_iris: List of right iris landmarks
            
        Returns:
            frame: Frame with eye landmarks drawn
        """
        if left_eye and right_eye:
            # Draw left eye landmarks
            for point in left_eye:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
            
            # Draw right eye landmarks
            for point in right_eye:
                cv2.circle(frame, point, 1, (0, 255, 0), -1)
            
            # Draw left iris landmarks
            for point in left_iris:
                cv2.circle(frame, point, 1, (255, 0, 0), -1)
            
            # Draw right iris landmarks
            for point in right_iris:
                cv2.circle(frame, point, 1, (255, 0, 0), -1)
        
        return frame 