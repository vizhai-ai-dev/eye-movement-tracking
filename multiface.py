import cv2
import mediapipe as mp
import numpy as np
import base64
from fastapi import FastAPI, UploadFile, File, WebSocket, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional
import json
from datetime import datetime
import os
import argparse
import asyncio
import threading
import queue
from pathlib import Path
from contextlib import asynccontextmanager

# Import eye tracking components
from gaze_tracker.detector import FaceDetector
from gaze_tracker.gaze_estimator import GazeEstimator
from gaze_tracker.tracker import GazeTracker

# Face detection setup
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(
    model_selection=1, 
    min_detection_confidence=0.2  
)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    min_detection_confidence=0.2,
    min_tracking_confidence=0.2
)

# Eye tracking setup
eye_detector = FaceDetector(min_detection_confidence=0.5)
gaze_estimator = GazeEstimator()
gaze_tracker = GazeTracker(suspicious_threshold=0.3, time_window=5)

# System state management
class SystemState:
    def __init__(self):
        self.face_detection_active = False
        self.eye_tracking_active = False
        self.calibration_in_progress = False
        self.is_calibrated = False
        self.suspicious_detected = False
        self.suspicious_start_time = None
        self.latest_gaze_data = {
            'gaze_direction': 'unknown',
            'off_screen_ratio': 0.0,
            'status': 'inactive',
            'is_calibrated': False,
            'calibration_in_progress': False
        }

system_state = SystemState()

# Shared resources
camera = None
frame_queue = queue.Queue()
stop_event = threading.Event()

class FaceDetectionResponse(BaseModel):
    face_count: int
    suspicious: bool
    faces: List[dict]
    annotated_image_base64: Optional[str] = None
    timestamp: str
    suspicious_duration: Optional[float] = None

class GazeTrackingResponse(BaseModel):
    gaze_direction: str
    off_screen_ratio: float
    status: str
    is_calibrated: bool
    calibration_in_progress: bool
    frame_base64: Optional[str] = None
    timestamp: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=camera_thread, daemon=True).start()
    yield
    stop_event.set()
    if camera is not None:
        camera.release()

app = FastAPI(title="AI Proctoring System", lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def process_frame_face_detection(frame):
    global system_state
    
    ih, iw, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(rgb_frame)
    mesh_results = face_mesh.process(rgb_frame)
    
    annotated_frame = frame.copy()
    faces = []
    
    if detection_results.detections:
        for detection in detection_results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                         int(bboxC.width * iw), int(bboxC.height * ih)
            
            x = max(0, x)
            y = max(0, y)
            w = min(iw - x, w)
            h = min(ih - y, h)
            
            confidence = detection.score[0]
            faces.append({
                "box": [x, y, w, h],
                "confidence": float(confidence),
                "type": "full" 
            })
    
    if mesh_results.multi_face_landmarks:
        for face_landmarks in mesh_results.multi_face_landmarks:
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
            
            is_new_face = True
            for face in faces:
                fx, fy, fw, fh = face["box"]
                if (abs(x_min - fx) < 50 and abs(y_min - fy) < 50):
                    is_new_face = False
                    break
            
            if is_new_face:
                visible_landmarks = sum(1 for landmark in face_landmarks.landmark 
                                     if 0 <= landmark.x <= 1 and 0 <= landmark.y <= 1)
                confidence = visible_landmarks / len(face_landmarks.landmark)
                
                if confidence > 0.2: 
                    faces.append({
                        "box": [x_min, y_min, x_max - x_min, y_max - y_min],
                        "confidence": float(confidence),
                        "type": "partial"
                    })
    
    for face in faces:
        x, y, w, h = face["box"]
        confidence = face["confidence"]
        
        color = (0, 255, 0) if len(faces) == 1 else (0, 0, 255)
        cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)
        
        label = f'{confidence:.2f} ({face["type"]})'
        cv2.putText(annotated_frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    if len(faces) > 1 or (len(faces) == 1 and faces[0]["confidence"] < 0.3):
        if not system_state.suspicious_detected:
            system_state.suspicious_detected = True
            system_state.suspicious_start_time = datetime.now()
            save_suspicious_frame(annotated_frame)
    else:
        system_state.suspicious_detected = False
        system_state.suspicious_start_time = None
    
    return faces, annotated_frame

def process_frame_eye_tracking(frame):
    global system_state
    
    if not system_state.eye_tracking_active:
        return frame, system_state.latest_gaze_data
    
    # Flip the frame horizontally for eye tracking
    frame = cv2.flip(frame, 1)
    
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
            'calibration_in_progress': False
        })
        
        # Draw visualizations
        frame = eye_detector.draw_eye_landmarks(frame, left_eye, right_eye, left_iris, right_iris)
        frame = gaze_estimator.draw_gaze_visualization(
            frame, left_eye, right_eye, left_iris, right_iris, gaze_direction
        )
        frame = gaze_tracker.draw_status_visualization(frame, gaze_direction)
    
    return frame, system_state.latest_gaze_data

def save_suspicious_frame(frame):
    os.makedirs("screenshots", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshots/suspicious_{timestamp}.jpg"
    cv2.imwrite(filename, frame)
    log_suspicious_event(timestamp, filename)

def log_suspicious_event(timestamp, filename):
    log_entry = {
        "timestamp": timestamp,
        "screenshot": filename,
        "type": "suspicious_activity"
    }
    os.makedirs("logs", exist_ok=True)
    with open("logs/suspicious_events.log", "a") as f:
        f.write(json.dumps(log_entry) + "\n")

def encode_image_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode('utf-8')

def camera_thread():
    global camera
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_FPS, 30)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    while not stop_event.is_set():
        ret, frame = camera.read()
        if ret:
            frame_queue.put(frame)
        else:
            break
    
    camera.release()

# Face Detection Routes
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect", response_model=FaceDetectionResponse)
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    faces, annotated_frame = process_frame_face_detection(frame)
    
    suspicious_duration = None
    if system_state.suspicious_detected and system_state.suspicious_start_time:
        suspicious_duration = (datetime.now() - system_state.suspicious_start_time).total_seconds()
    
    response = FaceDetectionResponse(
        face_count=len(faces),
        suspicious=len(faces) > 1 or (len(faces) == 1 and faces[0]["confidence"] < 0.3),
        faces=faces,
        annotated_image_base64=encode_image_to_base64(annotated_frame),
        timestamp=datetime.now().isoformat(),
        suspicious_duration=suspicious_duration
    )
    
    return response

@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                faces, processed_frame = process_frame_face_detection(frame)
                
                _, buffer = cv2.imencode('.jpg', processed_frame)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in video feed: {e}")
                break
    
    return StreamingResponse(
        generate(),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            try:
                frame = frame_queue.get(timeout=1)
                faces, processed_frame = process_frame_face_detection(frame)
                
                suspicious_duration = None
                if system_state.suspicious_detected and system_state.suspicious_start_time:
                    suspicious_duration = (datetime.now() - system_state.suspicious_start_time).total_seconds()
                
                await websocket.send_json({
                    "face_count": len(faces),
                    "suspicious": len(faces) > 1 or (len(faces) == 1 and faces[0]["confidence"] < 0.3),
                    "faces": faces,
                    "annotated_image_base64": encode_image_to_base64(processed_frame),
                    "timestamp": datetime.now().isoformat(),
                    "suspicious_duration": suspicious_duration
                })
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in websocket: {e}")
                break
    finally:
        await websocket.close()

# New unified API endpoints
@app.post("/api/system/start")
async def start_system(mode: str):
    """Start the system in specified mode: 'face', 'eye', or 'both'"""
    global system_state
    
    if mode not in ['face', 'eye', 'both']:
        raise HTTPException(status_code=400, detail="Invalid mode specified")
    
    if mode in ['face', 'both']:
        system_state.face_detection_active = True
    
    if mode in ['eye', 'both']:
        system_state.eye_tracking_active = True
        # Initialize eye tracking components
        eye_detector.reset()
        gaze_tracker.reset()
    
    return {"status": "started", "mode": mode}

@app.post("/api/system/stop")
async def stop_system(mode: str):
    """Stop the system in specified mode: 'face', 'eye', or 'both'"""
    global system_state
    
    if mode not in ['face', 'eye', 'both']:
        raise HTTPException(status_code=400, detail="Invalid mode specified")
    
    if mode in ['face', 'both']:
        system_state.face_detection_active = False
    
    if mode in ['eye', 'both']:
        system_state.eye_tracking_active = False
        system_state.latest_gaze_data['status'] = 'inactive'
    
    return {"status": "stopped", "mode": mode}

@app.get("/api/system/status")
async def get_system_status():
    """Get the current status of both face detection and eye tracking"""
    global system_state
    
    suspicious_duration = None
    if system_state.suspicious_detected and system_state.suspicious_start_time:
        suspicious_duration = (datetime.now() - system_state.suspicious_start_time).total_seconds()
    
    return {
        "face_detection": {
            "active": system_state.face_detection_active,
            "suspicious": system_state.suspicious_detected,
            "suspicious_duration": suspicious_duration
        },
        "eye_tracking": {
            "active": system_state.eye_tracking_active,
            "calibrated": system_state.is_calibrated,
            "calibrating": system_state.calibration_in_progress,
            "gaze_data": system_state.latest_gaze_data
        },
        "timestamp": datetime.now().isoformat()
    }

# Update existing endpoints to use system_state
@app.post("/api/start")
async def start_eye_tracking():
    return await start_system('eye')

@app.post("/api/stop")
async def stop_eye_tracking():
    return await stop_system('eye')

@app.get("/api/status")
async def get_eye_tracking_status():
    status = await get_system_status()
    return status["eye_tracking"]

def parse_args():
    parser = argparse.ArgumentParser(description='AI Proctoring System')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the server on (default: 8000)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the server on (default: 0.0.0.0)')
    return parser.parse_args()

if __name__ == "__main__":
    import uvicorn
    
    args = parse_args()
    port = int(os.getenv('PORT', args.port))
    host = os.getenv('HOST', args.host)
    
    print(f"Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port, ws_ping_interval=20, ws_ping_timeout=20) 