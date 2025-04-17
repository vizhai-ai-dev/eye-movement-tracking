# Eye Movement Tracking System (CLI Version)

This is the command-line interface version of the Eye Movement Tracking System. It provides a simple way to test the gaze tracking functionality without a web browser.

## Requirements

- Python 3.8+
- OpenCV
- Mediapipe
- Webcam access

## Setup

1. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the CLI application:
   ```
   python cli.py
   ```

## Usage

- The application will open a window showing the webcam feed with eye tracking overlay.
- It will display gaze direction information in the console.
- Press 'q' to exit the application.

## Features

- Real-time detection of face and eye landmarks
- Estimation of gaze direction (left, right, center, up, down)
- Calculation of off-screen time ratio
- Visual indicators for suspicious behavior
- Gaze heatmap overlay

## Example Output

```
Gaze: center    | Off-screen: 0.05 | Status: ok
```

When the user looks away from the screen for an extended period, the status will change to "suspicious".

## Troubleshooting

- If you get an error about the webcam not being available, make sure your webcam is connected and not being used by another application.
- If face detection is not working well, try adjusting the lighting conditions or camera position.
- For better performance, reduce `time_window` in the `GazeTracker` initialization if your machine is slower.