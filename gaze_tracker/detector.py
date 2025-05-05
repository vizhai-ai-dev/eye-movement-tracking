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
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Drawing specifications
        self.landmark_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),
            thickness=1,
            circle_radius=1
        )
        self.connection_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(255, 255, 255),
            thickness=1
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
        if frame is None:
            return None, None
            
        # Create a copy for visualization
        annotated_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # To improve performance, optionally mark the image as not writeable
        rgb_frame.flags.writeable = False
        
        # Process the frame and find face landmarks
        results = self.face_mesh.process(rgb_frame)
        
        # Enable writing to the image again
        rgb_frame.flags.writeable = True
        
        if not results.multi_face_landmarks:
            return None, annotated_frame
        
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert normalized landmarks to pixel coordinates
        landmarks = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            landmarks.append((x, y))
        
        # Draw the face mesh annotations
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=self.landmark_drawing_spec,
            connection_drawing_spec=self.connection_drawing_spec
        )
        
        # Draw eyes contour
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_LEFT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        self.mp_drawing.draw_landmarks(
            image=annotated_frame,
            landmark_list=face_landmarks,
            connections=self.mp_face_mesh.FACEMESH_RIGHT_EYE,
            landmark_drawing_spec=None,
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
        )
        
        return landmarks, annotated_frame

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
        
        try:
            left_eye = [landmarks[i] for i in self.LEFT_EYE_INDICES]
            right_eye = [landmarks[i] for i in self.RIGHT_EYE_INDICES]
            left_iris = [landmarks[i] for i in self.LEFT_IRIS_INDICES]
            right_iris = [landmarks[i] for i in self.RIGHT_IRIS_INDICES]
            
            return left_eye, right_eye, left_iris, right_iris
        except IndexError:
            print("Warning: Could not extract all eye landmarks")
            return None, None, None, None
    
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
        if frame is None:
            return None
            
        viz_frame = frame.copy()
        
        if left_eye and right_eye:
            # Draw eye contours
            cv2.polylines(viz_frame, [np.array(left_eye)], True, (0, 255, 0), 2)
            cv2.polylines(viz_frame, [np.array(right_eye)], True, (0, 255, 0), 2)
            
            # Draw iris contours and centers
            if left_iris and right_iris:
                # Left iris
                cv2.polylines(viz_frame, [np.array(left_iris)], True, (255, 0, 0), 2)
                left_center = np.mean(left_iris, axis=0).astype(int)
                cv2.circle(viz_frame, tuple(left_center), 2, (0, 0, 255), -1)
                
                # Right iris
                cv2.polylines(viz_frame, [np.array(right_iris)], True, (255, 0, 0), 2)
                right_center = np.mean(right_iris, axis=0).astype(int)
                cv2.circle(viz_frame, tuple(right_center), 2, (0, 0, 255), -1)
        
        return viz_frame 