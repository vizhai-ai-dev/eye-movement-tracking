// AI Proctoring System Client-side Application
document.addEventListener('DOMContentLoaded', function() {
    // Configuration
    const API_BASE_URL = '';  // Empty for same-origin requests
    const UPDATE_INTERVAL = 200;  // Update UI every 200ms
    
    // UI Elements - Face Detection
    const videoFeed = document.getElementById('video-feed');
    const faceStartButton = document.getElementById('face-start-button');
    const faceStopButton = document.getElementById('face-stop-button');
    const faceCount = document.getElementById('face-count');
    const faceStatus = document.getElementById('face-status');
    const suspiciousDuration = document.getElementById('suspicious-duration');
    
    // UI Elements - Eye Tracking
    const eyeStartButton = document.getElementById('eye-start-button');
    const eyeStopButton = document.getElementById('eye-stop-button');
    const calibrateButton = document.getElementById('calibrate-button');
    const trackingStatus = document.getElementById('tracking-status');
    const calibrationStatus = document.getElementById('calibration-status');
    const gazeDirection = document.getElementById('gaze-direction');
    const offScreenRatio = document.getElementById('off-screen-ratio');
    
    // System state
    const systemState = {
        faceDetectionActive: false,
        eyeTrackingActive: false,
        calibrationInProgress: false,
        wsConnection: null,
        statusUpdateTimer: null
    };
    
    // Initialize
    function init() {
        // Setup face detection buttons
        faceStartButton.addEventListener('click', () => startSystem('face'));
        faceStopButton.addEventListener('click', () => stopSystem('face'));
        
        // Setup eye tracking buttons
        eyeStartButton.addEventListener('click', () => startSystem('eye'));
        eyeStopButton.addEventListener('click', () => stopSystem('eye'));
        calibrateButton.addEventListener('click', startCalibration);
        
        // Initialize UI state
        updateFaceDetectionUI(false);
        updateEyeTrackingUI(false);
        
        // Start periodic status updates
        startStatusUpdates();
        
        // Add periodic video feed status check
        setInterval(checkVideoFeedStatus, 5000);
    }
    
    // System control functions
    async function startSystem(mode) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/system/start`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            
            if (!response.ok) throw new Error('Failed to start system');
            
            const result = await response.json();
            
            if (mode === 'face' || mode === 'both') {
                systemState.faceDetectionActive = true;
                initializeWebSocket();
            }
            
            if (mode === 'eye' || mode === 'both') {
                systemState.eyeTrackingActive = true;
            }
            
            updateSystemUI();
            showMessage(`System started in ${mode} mode`);
        } catch (error) {
            showError('Error starting system: ' + error.message);
        }
    }
    
    async function stopSystem(mode) {
        try {
            const response = await fetch(`${API_BASE_URL}/api/system/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            
            if (!response.ok) throw new Error('Failed to stop system');
            
            if (mode === 'face' || mode === 'both') {
                systemState.faceDetectionActive = false;
                if (systemState.wsConnection) {
                    systemState.wsConnection.close();
                    systemState.wsConnection = null;
                }
            }
            
            if (mode === 'eye' || mode === 'both') {
                systemState.eyeTrackingActive = false;
            }
            
            // Clear video feed if both systems are inactive
            if (!systemState.faceDetectionActive && !systemState.eyeTrackingActive) {
                clearVideoFeed();
            }
            
            updateSystemUI();
            showMessage(`System stopped in ${mode} mode`);
        } catch (error) {
            showError('Error stopping system: ' + error.message);
        }
    }
    
    function updateSystemUI() {
        updateFaceDetectionUI(systemState.faceDetectionActive);
        updateEyeTrackingUI(systemState.eyeTrackingActive);
        
        calibrateButton.disabled = !systemState.eyeTrackingActive;
    }
    
    // Status updates
    async function startStatusUpdates() {
        if (systemState.statusUpdateTimer === null) {
            updateSystemStatus();  // Update immediately
            systemState.statusUpdateTimer = setInterval(updateSystemStatus, UPDATE_INTERVAL);
        }
    }
    
    async function updateSystemStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/api/system/status?include_frame=${!systemState.wsConnection}`);
            if (!response.ok) throw new Error('Failed to get system status');
            
            const status = await response.json();
            
            // Update face detection metrics
            if (status.face_detection) {
                faceCount.textContent = status.face_detection.face_count || '0';
                faceStatus.textContent = status.face_detection.suspicious ? 'Suspicious' : 'Normal';
                faceStatus.className = 'status-value ' + (status.face_detection.suspicious ? 'suspicious' : 'normal');
                
                if (status.face_detection.suspicious_duration) {
                    suspiciousDuration.textContent = status.face_detection.suspicious_duration.toFixed(1) + 's';
                    suspiciousDuration.className = 'status-value suspicious';
                }
            }
            
            // Update eye tracking metrics
            if (status.eye_tracking) {
                const eyeData = status.eye_tracking;
                trackingStatus.textContent = eyeData.active ? 'Active' : 'Inactive';
                gazeDirection.textContent = eyeData.gaze_data.gaze_direction;
                offScreenRatio.textContent = (eyeData.gaze_data.off_screen_ratio * 100).toFixed(1) + '%';
                
                if (eyeData.calibrating) {
                    calibrationStatus.textContent = 'Calibrating...';
                    calibrationStatus.className = 'status-value calibrating';
                } else if (eyeData.calibrated) {
                    calibrationStatus.textContent = 'Calibrated';
                    calibrationStatus.className = 'status-value calibrated';
                } else {
                    calibrationStatus.textContent = 'Not calibrated';
                    calibrationStatus.className = 'status-value not-calibrated';
                }
                
                // Update video feed if WebSocket is not available
                if (!systemState.wsConnection && status.frame) {
                    updateVideoFeed(status.frame);
                }
            }
        } catch (error) {
            console.error('Error updating system status:', error);
        }
    }
    
    // WebSocket handling for face detection
    function initializeWebSocket() {
        if (systemState.wsConnection) return;
        
        try {
            const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${wsProtocol}//${window.location.host}/ws`;
            systemState.wsConnection = new WebSocket(wsUrl);
            
            systemState.wsConnection.onopen = function() {
                console.log('WebSocket connection established');
                showMessage('Connected to video feed');
                document.getElementById('video-placeholder').style.display = 'none';
            };
            
            systemState.wsConnection.onmessage = function(event) {
                const data = JSON.parse(event.data);
                
                // Update video feed
                if (data.frame) {
                    updateVideoFeed(data.frame);
                }
                
                // Update face detection metrics
                if (data.face_data) {
                    faceCount.textContent = data.face_data.face_count;
                    faceStatus.textContent = data.face_data.suspicious ? 'Suspicious' : 'Normal';
                    faceStatus.className = 'status-value ' + (data.face_data.suspicious ? 'suspicious' : 'normal');
                    
                    if (data.face_data.suspicious_duration) {
                        suspiciousDuration.textContent = data.face_data.suspicious_duration.toFixed(1) + 's';
                        suspiciousDuration.className = 'status-value suspicious';
                    }
                }
                
                // Update eye tracking metrics
                if (data.eye_data) {
                    trackingStatus.textContent = data.eye_data.status;
                    gazeDirection.textContent = data.eye_data.gaze_direction;
                    offScreenRatio.textContent = (data.eye_data.off_screen_ratio * 100).toFixed(1) + '%';
                    
                    if (data.eye_data.calibration_in_progress) {
                        calibrationStatus.textContent = 'Calibrating...';
                        calibrationStatus.className = 'status-value calibrating';
                    } else if (data.eye_data.is_calibrated) {
                        calibrationStatus.textContent = 'Calibrated';
                        calibrationStatus.className = 'status-value calibrated';
                    } else {
                        calibrationStatus.textContent = 'Not calibrated';
                        calibrationStatus.className = 'status-value not-calibrated';
                    }
                }
            };
            
            systemState.wsConnection.onclose = function() {
                console.log('WebSocket connection closed');
                systemState.wsConnection = null;
                // Attempt to reconnect after 2 seconds
                setTimeout(() => {
                    if (systemState.faceDetectionActive) {
                        initializeWebSocket();
                    }
                }, 2000);
            };
            
            systemState.wsConnection.onerror = function(error) {
                console.error('WebSocket error:', error);
                showError('Video feed connection error. Retrying...');
            };
        } catch (error) {
            console.error('Error initializing WebSocket:', error);
            showError('Failed to initialize video feed');
            // Fallback to HTTP polling
            startStatusUpdates();
        }
    }
    
    // UI Updates
    function updateFaceDetectionUI(active) {
        faceStartButton.disabled = active;
        faceStopButton.disabled = !active;
        
        if (!active) {
            faceCount.textContent = '0';
            faceStatus.textContent = 'Not running';
            faceStatus.className = 'status-value';
            suspiciousDuration.textContent = '0s';
            suspiciousDuration.className = 'status-value';
            if (!systemState.eyeTrackingActive) {
                videoFeed.src = '';
            }
        }
    }
    
    function updateEyeTrackingUI(active) {
        eyeStartButton.disabled = active;
        eyeStopButton.disabled = !active;
        calibrateButton.disabled = !active;
        
        if (!active) {
            trackingStatus.textContent = 'Inactive';
            gazeDirection.textContent = 'Unknown';
            offScreenRatio.textContent = '0%';
            calibrationStatus.textContent = 'Not calibrated';
            calibrationStatus.className = 'status-value not-calibrated';
            if (!systemState.faceDetectionActive) {
                videoFeed.src = '';
            }
        }
    }
    
    function updateVideoFeed(base64Image) {
        try {
            if (!videoFeed) return;
            
            // Hide placeholder if visible
            const placeholder = document.getElementById('video-placeholder');
            if (placeholder) {
                placeholder.style.display = 'none';
            }
            
            // Update video feed
            videoFeed.src = `data:image/jpeg;base64,${base64Image}`;
            videoFeed.classList.add('active');
            
            // Handle load error
            videoFeed.onerror = function() {
                console.error('Error loading video frame');
                videoFeed.classList.remove('active');
                if (placeholder) {
                    placeholder.style.display = 'block';
                    placeholder.textContent = 'Error loading video feed';
                }
            };
        } catch (error) {
            console.error('Error updating video feed:', error);
            showError('Error displaying video feed');
        }
    }
    
    function clearVideoFeed() {
        if (!videoFeed) return;
        
        videoFeed.src = '';
        videoFeed.classList.remove('active');
        
        const placeholder = document.getElementById('video-placeholder');
        if (placeholder) {
            placeholder.style.display = 'block';
            placeholder.textContent = 'Video feed will appear here';
        }
    }
    
    // Add new function to check video feed status
    function checkVideoFeedStatus() {
        if (!videoFeed) return;
        
        if (!videoFeed.src || videoFeed.src === 'data:,') {
            const placeholder = document.getElementById('video-placeholder');
            if (placeholder) {
                placeholder.style.display = 'block';
                if (systemState.faceDetectionActive || systemState.eyeTrackingActive) {
                    placeholder.textContent = 'Connecting to video feed...';
                } else {
                    placeholder.textContent = 'Video feed will appear here';
                }
            }
        }
    }
    
    // API Calls
    async function apiCall(endpoint, method = 'GET', data = null) {
        const options = {
            method: method,
            headers: {
                'Content-Type': 'application/json'
            }
        };
        
        if (data && (method === 'POST' || method === 'PUT')) {
            options.body = JSON.stringify(data);
        }
        
        try {
            const response = await fetch(`${API_BASE_URL}/api/${endpoint}`, options);
            if (!response.ok) {
                throw new Error(`API Error: ${response.status} ${response.statusText}`);
            }
            return await response.json();
        } catch (error) {
            console.error('API Call Failed:', error);
            showError(error.message);
            return null;
        }
    }
    
    // Message Display
    function showMessage(message) {
        const messageElement = document.getElementById('message');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.className = 'info-message';
            messageElement.style.display = 'block';
            setTimeout(() => {
                messageElement.style.display = 'none';
                messageElement.className = '';
            }, 3000);
        }
    }
    
    function showError(message) {
        const messageElement = document.getElementById('message');
        if (messageElement) {
            messageElement.textContent = message;
            messageElement.className = 'error-message';
            messageElement.style.display = 'block';
            setTimeout(() => {
                messageElement.style.display = 'none';
                messageElement.className = '';
            }, 5000);
        }
    }
    
    // Initialize the application
    init();
}); 
