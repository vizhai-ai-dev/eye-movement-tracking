import cv2
import numpy as np
import time
import base64
import json
import threading
from flask import Flask, jsonify, request, Response
from flask_cors import CORS
from gaze_tracker.detector import FaceDetector
from gaze_tracker.gaze_estimator import GazeEstimator
from gaze_tracker.tracker import GazeTracker

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize the webcam, face detector, gaze estimator, and gaze tracker
cap = None
face_detector = FaceDetector(min_detection_confidence=0.5)
gaze_estimator = GazeEstimator()
gaze_tracker = GazeTracker(suspicious_threshold=0.3, time_window=5)  # Reduce time window for faster response

# Global variables for storing the latest results
latest_frame = None
latest_frame_encoded = None
latest_gaze_direction = "unknown"
latest_off_screen_ratio = 0.0
latest_status = "ok"
latest_raw_data = {}
calibration_in_progress = False
calibration_start_time = None
frame_skip = 0  # Frame skipping for performance (0 = process every frame)

# Processing thread control
processing_active = False
processing_thread = None
thread_lock = threading.Lock()

def initialize_camera():
    """Initialize the webcam"""
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Use default webcam (change index if needed)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS for smoother processing
    return cap.isOpened()

def encode_frame(frame):
    """Encode a frame as base64 for sending over API"""
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

def process_frame():
    """Process a single frame from the webcam"""
    global latest_frame, latest_frame_encoded, latest_gaze_direction, latest_off_screen_ratio, latest_status, latest_raw_data
    global cap, calibration_in_progress, calibration_start_time, frame_skip
    
    if cap is None or not cap.isOpened():
        if not initialize_camera():
            print("Error: Could not initialize webcam")
            return False
    
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam")
        return False
    
    # Flip the frame horizontally (mirror view)
    frame = cv2.flip(frame, 1)
    
    # Skip frame processing for performance if needed
    if frame_skip > 0:
        frame_skip = (frame_skip + 1) % 2  # Skip every other frame
        latest_frame = frame  # Still update the display
        latest_frame_encoded = encode_frame(frame)
        return True
    
    # Process this frame
    
    # Detect face landmarks
    landmarks, _ = face_detector.detect_face_landmarks(frame)
    
    # Get eye landmarks
    left_eye, right_eye, left_iris, right_iris = face_detector.get_eye_landmarks(landmarks)
    
    # Handle calibration if in progress
    if calibration_in_progress:
        if calibration_start_time is None:
            calibration_start_time = time.time()
            gaze_estimator.reset_calibration()
            
        # Add calibration sample
        calibration_complete = gaze_estimator.add_calibration_sample(left_eye, right_eye, left_iris, right_iris)
        
        # If calibration complete or timed out after 5 seconds
        elapsed_time = time.time() - calibration_start_time
        if calibration_complete or elapsed_time > 5:
            calibration_in_progress = False
            calibration_start_time = None
            if not calibration_complete:
                gaze_estimator._finalize_calibration()  # Force finalize if timed out
    
    # Estimate gaze direction
    gaze_direction, is_looking_at_screen, ratios = gaze_estimator.estimate_gaze_direction(
        left_eye, right_eye, left_iris, right_iris
    )
    
    # Update gaze tracker
    off_screen_ratio, status, _ = gaze_tracker.update(
        gaze_direction, is_looking_at_screen, ratios
    )
    
    # Create a clean copy of the frame for API consumers
    clean_frame = frame.copy()
    
    # Update global variables
    latest_frame = clean_frame
    latest_frame_encoded = encode_frame(clean_frame)
    latest_gaze_direction = gaze_direction
    latest_off_screen_ratio = off_screen_ratio
    latest_status = status
    
    # Store raw data for detailed API responses
    latest_raw_data = {
        'gaze_direction': gaze_direction,
        'is_looking_at_screen': is_looking_at_screen,
        'off_screen_ratio': off_screen_ratio,
        'status': status,
        'ratios': ratios,
        'calibration': {
            'is_calibrated': gaze_estimator.is_calibrated,
            'in_progress': calibration_in_progress,
            'needed': gaze_estimator.calibration_needed,
            'center_h': gaze_estimator.center_horizontal_ratio,
            'center_v': gaze_estimator.center_vertical_ratio,
            'h_threshold': gaze_estimator.horizontal_ratio_threshold,
            'v_threshold': gaze_estimator.vertical_ratio_threshold
        },
        'consecutive_off_screen': gaze_tracker.consecutive_off_screen
    }
    
    return True

def processing_loop():
    """Main processing loop that runs in a separate thread"""
    global processing_active
    
    print("Starting processing loop")
    while processing_active:
        process_frame()
        time.sleep(0.02)  # ~50 FPS

    print("Processing loop stopped")

@app.route('/api/start', methods=['POST'])
def start_processing():
    """Start the gaze tracking processing"""
    global processing_active, processing_thread
    
    if processing_thread is not None and processing_thread.is_alive():
        return jsonify({'status': 'already_running', 'message': 'Processing is already running'})
    
    # Initialize camera
    if not initialize_camera():
        return jsonify({'status': 'error', 'message': 'Failed to initialize camera'})
    
    # Start processing thread
    with thread_lock:
        processing_active = True
        processing_thread = threading.Thread(target=processing_loop)
        processing_thread.daemon = True
        processing_thread.start()
    
    return jsonify({'status': 'started', 'message': 'Gaze tracking started'})

@app.route('/api/stop', methods=['POST'])
def stop_processing():
    """Stop the gaze tracking processing"""
    global processing_active, processing_thread
    
    with thread_lock:
        processing_active = False
    
    if processing_thread is not None:
        processing_thread.join(timeout=1.0)
        processing_thread = None
    
    return jsonify({'status': 'stopped', 'message': 'Gaze tracking stopped'})

@app.route('/api/status')
def get_status():
    """Get the current status of the gaze tracking system"""
    global latest_gaze_direction, latest_off_screen_ratio, latest_status, latest_raw_data
    
    include_frame = request.args.get('include_frame', 'false').lower() == 'true'
    
    response = {
        'running': processing_active,
        'gaze_direction': latest_gaze_direction,
        'off_screen_ratio': latest_off_screen_ratio,
        'status': latest_status,
        'calibration_needed': gaze_estimator.calibration_needed,
        'calibration_in_progress': calibration_in_progress,
        'is_calibrated': gaze_estimator.is_calibrated,
        'timestamp': time.time()
    }
    
    if include_frame and latest_frame_encoded is not None:
        response['frame'] = latest_frame_encoded
    
    return jsonify(response)

@app.route('/api/raw_data')
def get_raw_data():
    """Get detailed raw data from the gaze tracking system"""
    global latest_raw_data
    
    include_frame = request.args.get('include_frame', 'false').lower() == 'true'
    
    response = latest_raw_data.copy()
    response['timestamp'] = time.time()
    
    if include_frame and latest_frame_encoded is not None:
        response['frame'] = latest_frame_encoded
    
    return jsonify(response)

@app.route('/api/frame')
def get_frame():
    """Get the latest processed frame as JPEG image"""
    global latest_frame
    
    if latest_frame is None:
        return Response('No frame available', status=404)
    
    _, buffer = cv2.imencode('.jpg', latest_frame)
    response = Response(buffer.tobytes(), mimetype='image/jpeg')
    return response

@app.route('/api/start_calibration', methods=['POST'])
def start_calibration():
    """Start the calibration process"""
    global calibration_in_progress
    
    if not processing_active:
        return jsonify({'status': 'error', 'message': 'Processing must be started first'})
    
    calibration_in_progress = True
    return jsonify({'status': 'started', 'message': 'Calibration started'})

@app.route('/api/reset_calibration', methods=['POST'])
def reset_calibration():
    """Reset the calibration"""
    global calibration_in_progress
    
    calibration_in_progress = False
    gaze_estimator.reset_calibration()
    return jsonify({'status': 'reset', 'message': 'Calibration reset'})

@app.route('/api/settings', methods=['GET', 'POST'])
def update_settings():
    """Get or update system settings"""
    global frame_skip
    
    if request.method == 'POST':
        data = request.json
        
        if 'frame_skip' in data:
            frame_skip = int(data['frame_skip'])
        
        if 'suspicious_threshold' in data:
            threshold = float(data['suspicious_threshold'])
            gaze_tracker.suspicious_threshold = max(0.1, min(0.9, threshold))
            
        if 'time_window' in data:
            time_window = float(data['time_window'])
            gaze_tracker.time_window = max(1, min(30, time_window))
    
    # Return current settings
    return jsonify({
        'frame_skip': frame_skip,
        'suspicious_threshold': gaze_tracker.suspicious_threshold,
        'time_window': gaze_tracker.time_window,
        'horizontal_threshold': gaze_estimator.horizontal_ratio_threshold,
        'vertical_threshold': gaze_estimator.vertical_ratio_threshold
    })

def release_camera():
    """Release the webcam"""
    global cap
    if cap is not None and cap.isOpened():
        cap.release()
        cap = None

@app.route('/api/shutdown', methods=['POST'])
def shutdown():
    """Release the webcam and prepare for shutdown"""
    stop_processing()
    release_camera()
    return jsonify({'status': 'shutdown', 'message': 'Camera released and system shutdown'})

@app.route('/api')
def api_documentation():
    """Return API documentation"""
    docs = {
        'api_version': '1.0',
        'description': 'Eye Movement Tracking API',
        'endpoints': [
            {
                'path': '/api/start',
                'method': 'POST',
                'description': 'Start gaze tracking processing'
            },
            {
                'path': '/api/stop',
                'method': 'POST',
                'description': 'Stop gaze tracking processing'
            },
            {
                'path': '/api/status',
                'method': 'GET',
                'description': 'Get current gaze tracking status',
                'params': [
                    {'name': 'include_frame', 'type': 'boolean', 'description': 'Include base64 encoded frame in response'}
                ]
            },
            {
                'path': '/api/raw_data',
                'method': 'GET',
                'description': 'Get detailed raw data from gaze tracking',
                'params': [
                    {'name': 'include_frame', 'type': 'boolean', 'description': 'Include base64 encoded frame in response'}
                ]
            },
            {
                'path': '/api/frame',
                'method': 'GET',
                'description': 'Get latest processed frame as JPEG image'
            },
            {
                'path': '/api/start_calibration',
                'method': 'POST',
                'description': 'Start calibration process'
            },
            {
                'path': '/api/reset_calibration',
                'method': 'POST',
                'description': 'Reset calibration data'
            },
            {
                'path': '/api/settings',
                'method': 'GET/POST',
                'description': 'Get or update system settings'
            },
            {
                'path': '/api/shutdown',
                'method': 'POST',
                'description': 'Shutdown the system and release resources'
            }
        ]
    }
    
    return jsonify(docs)

@app.route('/')
def root():
    """Root endpoint redirects to API documentation"""
    return jsonify({
        'message': 'Eye Movement Tracking API Server',
        'documentation': '/api'
    })

if __name__ == '__main__':
    try:
        print("Starting Eye Movement Tracking API Server")
        print("Press Ctrl+C to exit")
        
        # Do NOT start processing by default to save resources
        # Client must explicitly call /api/start endpoint
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Stop processing and release the camera when the application is closed
        stop_processing()
        release_camera()
        print("Resources released. Goodbye!") 