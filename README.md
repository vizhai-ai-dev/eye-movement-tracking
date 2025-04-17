# Eye Movement Tracking API for Online Interviews

A real-time eye movement detection system that tracks a candidate's gaze direction via webcam feed and flags suspicious behavior during online interviews. This system is designed as a REST API that can be integrated into any web application.

## Features

- Real-time eye movement detection through API endpoints
- Gaze direction tracking (left, right, center, up, down)
- Off-screen time ratio calculation
- Suspicious behavior flagging
- Calibration for improved accuracy

## Requirements

- Python 3.8+
- OpenCV
- Mediapipe
- Flask
- Flask-CORS
- NumPy
- Pillow
- Webcam access

## Project Structure

- `app.py`: Main API entry point and server implementation
- `gaze_tracker/`: Core eye tracking functionality
  - `detector.py`: Face and eye landmark detection
  - `gaze_estimator.py`: Gaze direction estimation
  - `tracker.py`: Tracking gaze over time
- `static/`: Client-side example
- `templates/`: Client-side example HTML
- `requirements.txt`: Dependencies

## API Endpoints

- `/api/start` (POST): Start eye tracking processing
- `/api/stop` (POST): Stop eye tracking processing
- `/api/status` (GET): Get current tracking status
- `/api/raw_data` (GET): Get detailed tracking data
- `/api/frame` (GET): Get current webcam frame as JPEG
- `/api/start_calibration` (POST): Start calibration procedure
- `/api/reset_calibration` (POST): Reset calibration data
- `/api/settings` (GET/POST): Get or update system settings
- `/api/shutdown` (POST): Shutdown the system
- `/api` (GET): API documentation

## Setup

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the API server: `python app.py`
4. The API will be available at `http://localhost:8080/api`

## Integration

This API server can be integrated with any web application by making HTTP requests to the provided endpoints. All data is returned in JSON format, and images are base64 encoded when included in responses.

Example:
```javascript
// Start tracking
fetch('http://localhost:8080/api/start', { method: 'POST' })
  .then(response => response.json())
  .then(data => console.log(data));

// Get status
fetch('http://localhost:8080/api/status')
  .then(response => response.json())
  .then(data => {
    console.log(`Looking: ${data.gaze_direction}`);
    console.log(`Off-screen ratio: ${data.off_screen_ratio}`);
    console.log(`Status: ${data.status}`);
  });
```

## How It Works

The system uses a pre-trained gaze estimation model to detect eye movements and determine gaze direction. It processes webcam frames, estimates where the user is looking, and flags suspicious behavior if the user looks away from the screen for extended periods. 