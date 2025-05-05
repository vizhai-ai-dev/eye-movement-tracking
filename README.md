# AI Proctoring System with Multiface Detection

An advanced AI-powered proctoring system that combines multiface detection and intelligent eye tracking to monitor exam-taking behavior. The system is designed to be fair and practical, allowing natural behaviors like looking down at notes while detecting suspicious activities.

## Key Features

### Advanced Face Detection
- **Multiface Detection**: Detect and track multiple faces simultaneously
- **Face Mesh Visualization**: Detailed facial landmark tracking
- **Confidence Scoring**: Real-time confidence metrics for face detection
- **Suspicious Activity Detection**: Automatic flagging of multiple faces or unusual behavior
- **Automated Screenshots**: Capture and log suspicious events

### Intelligent Eye Tracking
- **Precise Iris Detection**: Advanced iris landmark tracking
- **Smart Gaze Analysis**: 
  - Asymmetric thresholds for different gaze directions
  - More lenient for natural behaviors (looking down at notes)
  - Stricter for suspicious behaviors (looking up/sideways)
- **Real-time Gaze Heatmap**: Visual representation of gaze patterns
- **Adaptive Calibration**: Self-adjusting thresholds based on user behavior

### Real-time Web Interface
- **Live Video Feed**: Real-time monitoring with WebSocket support
- **Dynamic Status Updates**: Instant feedback on detection status
- **Interactive Calibration**: User-friendly calibration process
- **Visual Analytics**: 
  - Gaze direction indicators
  - Off-screen time tracking
  - Suspicious behavior alerts
  - Status visualization

## Technical Stack

- **Backend**:
  - Python 3.8+
  - Flask (Web framework)
  - Flask-Sock (WebSocket support)
  - MediaPipe (Face mesh and iris detection)
  - OpenCV (Image processing)
  - NumPy (Numerical computations)

- **Frontend**:
  - HTML5/CSS3
  - JavaScript (Real-time updates)
  - WebSocket (Low-latency communication)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-proctoring-system.git
cd ai-proctoring-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Web Interface
1. Start the application:
```bash
python app.py
```

2. Open a web browser and navigate to:
```
http://localhost:8080
```

3. Use the interface buttons to:
   - Start/Stop Face Detection
   - Start/Stop Eye Tracking
   - Perform Calibration
   - View Analytics

### Command Line Interface
For automated testing or integration:
```bash
python cli.py --mode face  # For face detection only
python cli.py --mode eye   # For eye tracking only
python cli.py --mode both  # For both features
```

## Configuration

### Face Detection Settings
```python
FACE_DETECTION_CONFIDENCE = 0.5  # Detection confidence threshold
MAX_FACES = 2                    # Maximum number of faces to track
SUSPICIOUS_THRESHOLD = 0.3       # Threshold for suspicious behavior
```

### Eye Tracking Settings
```python
VERTICAL_UP_THRESHOLD = 0.30     # Threshold for upward gaze
VERTICAL_DOWN_THRESHOLD = 0.45   # More lenient threshold for downward gaze
HORIZONTAL_THRESHOLD = 0.25      # Threshold for left/right gaze
```

### Behavior Weights
```python
DIRECTION_WEIGHTS = {
    "center": 0.0,    # On-screen
    "up": 1.0,        # Off-screen
    "down": 0.2,      # Mostly on-screen (reading/writing)
    "left": 1.0,      # Off-screen
    "right": 1.0,     # Off-screen
    "down-left": 0.5, # Partially off-screen
    "down-right": 0.5 # Partially off-screen
}
```

## Docker Support

1. Build the container:
```bash
docker-compose build
```

2. Run the application:
```bash
docker-compose up
```

## Development

### Project Structure
```
.
├── app.py              # Main Flask application
├── multiface.py        # Multiface detection implementation
├── cli.py             # Command line interface
├── gaze_tracker/      # Core eye tracking module
│   ├── detector.py    # Face/eye detection
│   ├── estimator.py   # Gaze estimation
│   └── tracker.py     # Gaze tracking
├── templates/         # HTML templates
├── static/           # Static assets
└── tests/           # Test suite
```

### Running Tests
```bash
python -m pytest test_gaze_detector.py
```

## Contributing

1. Fork the repository
2. Create your feature branch: `git checkout -b feature/amazing-feature`
3. Commit your changes: `git commit -m 'Add amazing feature'`
4. Push to the branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- MediaPipe team for face mesh and iris detection
- OpenCV community for image processing tools
- Flask team for the web framework
- All contributors to this project 