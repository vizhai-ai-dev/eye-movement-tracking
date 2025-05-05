import cv2
import numpy as np
import time
import base64
import json
import threading
import mediapipe as mp
from flask import Flask, jsonify, request, Response, render_template
from flask_cors import CORS
from gaze_tracker.detector import FaceDetector
from gaze_tracker.gaze_estimator import GazeEstimator
from gaze_tracker.tracker import GazeTracker
from datetime import datetime
import os
from flask_sock import Sock

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
sock = Sock(app)  # Initialize WebSocket support

# Face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1,  # 0 for short-range, 1 for full-range
    min_detection_confidence=0.5  # Increased from 0.2 for better accuracy
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.5,  # Increased from 0.2
    min_tracking_confidence=0.5,   # Increased from 0.2
    refine_landmarks=True  # Enable landmark refinement for better accuracy
)

# Eye tracking setup
eye_detector = FaceDetector(min_detection_confidence=0.5)
gaze_estimator = GazeEstimator()
gaze_tracker = GazeTracker(suspicious_threshold=0.3, time_window=5)

# System state
class SystemState:
    def __init__(self):
        self.face_detection_active = False
        self.eye_tracking_active = False
        self.calibration_in_progress = False
        self.is_calibrated = False
        self.suspicious_detected = False
        self.suspicious_start_time = None
        self.latest_frame = None
        self.latest_frame_encoded = None
        self.latest_gaze_data = {
            'gaze_direction': 'unknown',
            'off_screen_ratio': 0.0,
            'status': 'inactive',
            'is_calibrated': False,
            'calibration_in_progress': False
        }
        self.face_data = {
            'face_count': 0,
            'suspicious': False,
            'faces': [],
            'suspicious_duration': None
        }

system_state = SystemState()

# Camera and processing setup
cap = None
processing_thread = None
thread_lock = threading.Lock()
processing_active = False

def initialize_camera():
    """Initialize the webcam"""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    return cap.isOpened()

def encode_frame(frame):
    """Encode a frame as base64 for sending over API"""
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def process_frame_face_detection(frame):
    """Process frame for face detection"""
    global system_state
    
    if frame is None or frame.size == 0:
        return frame
    
    # Ensure frame dimensions are valid
    ih, iw = frame.shape[:2]
    if ih == 0 or iw == 0:
        return frame
    
    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process with MediaPipe
    detection_results = face_detection.process(rgb_frame)
    mesh_results = face_mesh.process(rgb_frame)
    
    # Create a copy for annotations
    annotated_frame = frame.copy()
    faces = []
    
    # Process face detection results
    if detection_results.detections:
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            
            # Convert relative coordinates to absolute
            x = max(0, int(bboxC.xmin * iw))
            y = max(0, int(bboxC.ymin * ih))
            w = min(int(bboxC.width * iw), iw - x)
            h = min(int(bboxC.height * ih), ih - y)
            
            confidence = detection.score[0]
            faces.append({
                "box": [x, y, w, h],
                "confidence": float(confidence),
                "type": "full"
            })
            
            # Draw detection box
            color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
            
            # Add confidence label
            label = f'{confidence:.2f}'
            cv2.putText(annotated_frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Process face mesh results
    if mesh_results.multi_face_landmarks:
        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        
        for face_landmarks in mesh_results.multi_face_landmarks:
            # Calculate face mesh bounding box
            x_min = iw
            y_min = ih
            x_max = 0
            y_max = 0
            
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * iw), int(landmark.y * ih)
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x)
                y_max = max(y_max, y)
            
            # Check if this is a new face (not already detected)
            is_new_face = True
            for face in faces:
                fx, fy, fw, fh = face["box"]
                if (abs(x_min - fx) < 50 and abs(y_min - fy) < 50):
                    is_new_face = False
                    break
            
            if is_new_face:
                # Count visible landmarks
                visible_landmarks = sum(1 for landmark in face_landmarks.landmark 
                                     if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                confidence = visible_landmarks / len(face_landmarks.landmark)
                
                if confidence > 0.5:  # Increased threshold
                    faces.append({
                        "box": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "confidence": float(confidence),
                        "type": "partial"
                    })
            
            # Draw face mesh
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
    
    # Update system state
    is_suspicious = len(faces) > 1 or (len(faces) == 1 and faces[0]["confidence"] < 0.5)
    
    if is_suspicious:
        if not system_state.suspicious_detected:
            system_state.suspicious_detected = True
            system_state.suspicious_start_time = datetime.now()
            save_suspicious_frame(annotated_frame)
    else:
        system_state.suspicious_detected = False
        system_state.suspicious_start_time = None
    
    system_state.face_data.update({
        'face_count': len(faces),
        'suspicious': is_suspicious,
        'faces': faces,
        'suspicious_duration': (datetime.now() - system_state.suspicious_start_time).total_seconds() if system_state.suspicious_detected else None
    })
    
    return annotated_frame

def process_frame_eye_tracking(frame):
    """Process frame for eye tracking"""
    global system_state
    
    if not system_state.eye_tracking_active:
        return frame
    
    # Flip the frame horizontally for eye tracking
    frame = cv2.flip(frame, 1)
    
    # Create a copy for visualization
    annotated_frame = frame.copy()
    
    # Detect face landmarks
    landmarks, _ = eye_detector.detect_face_landmarks(frame)
    
    if landmarks:
        # Get eye landmarks
        left_eye, right_eye, left_iris, right_iris = eye_detector.get_eye_landmarks(landmarks)
        
        # Estimate gaze direction
        gaze_direction, is_looking_at_screen, ratios = gaze_estimator.estimate_gaze_direction(
            left_eye, right_eye, left_iris, right_iris
        )
        
        # Update gaze tracker
        off_screen_ratio, status, _ = gaze_tracker.update(
            gaze_direction, is_looking_at_screen, ratios
        )
        
        # Update latest gaze data
        system_state.latest_gaze_data.update({
            'gaze_direction': gaze_direction,
            'off_screen_ratio': off_screen_ratio,
            'status': status,
            'is_calibrated': gaze_estimator.is_calibrated,
            'calibration_in_progress': system_state.calibration_in_progress
        })
        
        # Draw visualizations
        annotated_frame = eye_detector.draw_eye_landmarks(annotated_frame, left_eye, right_eye, left_iris, right_iris)
        annotated_frame = gaze_estimator.draw_gaze_visualization(
            annotated_frame, left_eye, right_eye, left_iris, right_iris, gaze_direction
        )
        annotated_frame = gaze_tracker.draw_status_visualization(annotated_frame, gaze_direction)
        
        # Add status text
        cv2.putText(annotated_frame, f"Gaze: {gaze_direction}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f"Off-screen: {off_screen_ratio:.1%}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return annotated_frame

def save_suspicious_frame(frame):
    """Save suspicious activity screenshots"""
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/suspicious_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    log_suspicious_event(timestamp, filename)

def log_suspicious_event(timestamp, filename):
    """Log suspicious events"""
    log_entry = {
        "timestamp": timestamp,
        "screenshot": filename,
        "type": "suspicious_activity"
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/suspicious_events.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def process_frame():
    """Process a single frame for both face detection and eye tracking"""
    global system_state
    
    if cap is None or not cap.isOpened():
        if not initialize_camera():
            print("Error: Could not initialize webcam")
            return False
    
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        return False
    
    processed_frame = frame.copy()
    
    # Process face detection if active
    if system_state.face_detection_active:
        processed_frame = process_frame_face_detection(processed_frame)
    
    # Process eye tracking if active
    if system_state.eye_tracking_active:
        processed_frame = process_frame_eye_tracking(processed_frame)
    
    # Update latest frame
    system_state.latest_frame = processed_frame
    system_state.latest_frame_encoded = encode_frame(processed_frame)
    
    return True

def processing_loop():
    """Main processing loop"""
    global processing_active
    
    print("Starting processing loop")
    while processing_active:
        process_frame()
        time.sleep(0.02)  # ~50 FPS
    
    print("Processing loop stopped")

# Routes
@app.route('/')
def index():
    """Serve the main application page"""
    return render_template('index.html')

@app.route('/api/system/start', methods=['POST'])
def start_system():
    """Start the system in specified mode"""
    global processing_active, processing_thread, system_state
    
    data = request.get_json()
    mode = data.get('mode', 'both')
    
    if mode not in ['face', 'eye', 'both']:
        return jsonify({'status': 'error', 'message': 'Invalid mode specified'})
    
    if processing_thread is not None and processing_thread.is_alive():
        return jsonify({'status': 'already_running', 'message': 'System is already running'})
    
    # Initialize camera
    if not initialize_camera():
        return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})
    
    # Update system state
    if mode in ['face', 'both']:
        system_state.face_detection_active = True
    
    if mode in ['eye', 'both']:
        system_state.eye_tracking_active = True
    
    # Start processing thread
    with thread_lock:
        processing_active = True
        processing_thread = threading.Thread(target=processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
    
    return jsonify({'status': 'started', 'mode': mode})

@app.route('/api/system/stop', methods=['POST'])
def stop_system():
    """Stop the system in specified mode"""
    global processing_active, processing_thread, system_state
    
    data = request.get_json()
    mode = data.get('mode', 'both')
    
    if mode not in ['face', 'eye', 'both']:
        return jsonify({'status': 'error', 'message': 'Invalid mode specified'})
    
    if mode in ['face', 'both']:
        system_state.face_detection_active = False
    
    if mode in ['eye', 'both']:
        system_state.eye_tracking_active = False
        system_state.latest_gaze_data['status'] = 'inactive'
    
    # Stop processing if both systems are inactive
    if not system_state.face_detection_active and not system_state.eye_tracking_active:
        with thread_lock:
            processing_active = False
        
        if processing_thread is not None:
            processing_thread.join(timeout=1.0)
            processing_thread = None
    
    return jsonify({'status': 'stopped', 'mode': mode})

@app.route('/api/system/status')
def get_system_status():
    """Get the current status of both systems"""
    include_frame = request.args.get('include_frame', 'false').lower() == 'true'
    
    status = {
        'face_detection': {
            'active': system_state.face_detection_active,
            'suspicious': system_state.suspicious_detected,
            'face_count': system_state.face_data['face_count'],
            'suspicious_duration': system_state.face_data['suspicious_duration']
        },
        'eye_tracking': {
            'active': system_state.eye_tracking_active,
            'calibrated': system_state.is_calibrated,
            'calibrating': system_state.calibration_in_progress,
            'gaze_data': system_state.latest_gaze_data
        },
        'timestamp': datetime.now().isoformat()
    }
    
    if include_frame and system_state.latest_frame_encoded:
        status['frame'] = system_state.latest_frame_encoded
    
    return jsonify(status)

@app.route('/api/start_calibration', methods=['POST'])
def start_calibration():
    """Start eye tracking calibration"""
    global system_state
    
    if not system_state.eye_tracking_active:
        return jsonify({'status': 'error', 'message': 'Eye tracking must be started first'})
    
    system_state.calibration_in_progress = True
    gaze_estimator.reset_calibration()
    
    return jsonify({'status': 'started', 'message': 'Calibration started'})

@app.route('/api/reset_calibration', methods=['POST'])
def reset_calibration():
    """Reset eye tracking calibration"""
    global system_state
    
    gaze_estimator.reset_calibration()
    system_state.is_calibrated = False
    system_state.calibration_in_progress = False
    
    return jsonify({'status': 'reset', 'message': 'Calibration reset'})

# Add WebSocket endpoint for video feed
@sock.route('/ws')
def websocket(ws):
    """WebSocket endpoint for real-time video feed"""
    try:
        while True:
            if system_state.latest_frame_encoded:
                message = {
                    'frame': system_state.latest_frame_encoded,
                    'face_data': system_state.face_data,
                    'eye_data': system_state.latest_gaze_data,
                    'timestamp': datetime.now().isoformat()
                }
                ws.send(json.dumps(message))
                time.sleep(0.033)  # ~30 FPS
    except Exception as e:
        print(f"WebSocket error: {e}")

# Add a video feed endpoint as fallback
@app.route('/video_feed')
def video_feed():
    """Video feed endpoint (fallback for browsers not supporting WebSocket)"""
    def generate():
        while True:
            if system_state.latest_frame is not None:
                _, buffer = cv2.imencode('.jpg', system_state.latest_frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)  # ~30 FPS
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def cleanup():
    """Cleanup resources"""
    global cap
    
    if cap is not None:
        cap.release()
        cap = None

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=8080, threaded=True)
    finally:
        cleanup() 