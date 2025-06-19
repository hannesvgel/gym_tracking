# Flask application for physical exercise classification
# Uses an LSTM model

# Ngrok configuration for public tunnel (commented)
# .\ngrok.exe authtoken 2yW0RevuJWjwymKTw4mCVGBks0M_F3mDrSRCWZ9dJn5wGKYt
# .\ngrok.exe http http://localhost:8000

# IMPORTS
from flask import Flask, request, jsonify, send_from_directory
import webbrowser
import os
import json
import threading
import time
import numpy as np
from flask_cors import CORS

# FLASK CONFIGURATION
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS to allow requests from different domains

# LSTM MODEL CONFIGURATION
MODEL_AVAILABLE = False  # Flag to check if the model is available
model = None  # Global variable to contain the loaded model

# Exercise classes recognized by the model
CLASSES = ["bench_press", "squat", "lat_machine", "pull_up", "push_up", "split_squat"]

# Model parameters
SEQUENCE_LENGTH = 30    # Number of frames needed for a prediction
KEYPOINT_DIM = 132     # Keypoint dimension (33 landmarks * 4 coordinates: x, y, z, visibility)

# MODEL LOADING
# Attempt to load TensorFlow model
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
    print("TensorFlow imported successfully")
    
    # Check model file existence
    if os.path.exists('skeleton_lstm_multiclass6.h5'):
        model = load_model('skeleton_lstm_multiclass6.h5')
        print("Model loaded successfully")
        print(f"Model input shape: {model.input_shape}")
    else:
        print("Model file not found")
        MODEL_AVAILABLE = False
        
except ImportError as e:
    print(f"Import error: {e}")
    MODEL_AVAILABLE = False
except Exception as e:
    print(f"Error loading model: {e}")
    MODEL_AVAILABLE = False

# FLASK ROUTES

@app.route('/')
def index():
    """
    Main route that serves the application's HTML file
    """
    return send_from_directory('static', 'index.html')

@app.route('/classify', methods=['POST'])
def classify():
    """
    Endpoint for exercise classification
    
    Receives:
        - JSON with sequence of landmarks (30 frames of poses)
    
    Returns:
        - JSON with exercise prediction, confidence and class name
    """
    
    # Check model availability
    if not MODEL_AVAILABLE or model is None:
        return jsonify({
            'error': 'Model not available',
            'exercise': None,
            'confidence': 0,
            'class_name': None
        }), 503

    try:
        # Receive data from index.html and extract data from request
        data = request.get_json()
        sequence = data['sequence']
        landmarks_sequence = np.array(sequence, dtype=np.float32)

        # SEQUENCE DIMENSION VALIDATION
        
        # Check sequence length (must be 30 frames)
        if landmarks_sequence.shape[0] != SEQUENCE_LENGTH:
            return jsonify({
                'error': f'Invalid sequence length. Expected {SEQUENCE_LENGTH}, got {landmarks_sequence.shape[0]}',
                'exercise': None,
                'confidence': 0,
                'class_name': None
            }), 400

        # Check dimensions of each frame
        expected_frame_size = KEYPOINT_DIM
        if landmarks_sequence.shape[1] != expected_frame_size:
            # Attempt to reshape if data is in correct format but with different shape
            if landmarks_sequence.size == SEQUENCE_LENGTH * expected_frame_size:
                landmarks_sequence = landmarks_sequence.reshape(SEQUENCE_LENGTH, expected_frame_size)
            else:
                return jsonify({
                    'error': f'Invalid frame size. Each frame should have {expected_frame_size} values',
                    'exercise': None,
                    'confidence': 0,
                    'class_name': None
                }), 400

        # PREDICTION
        
        # Prepare input for the model
        input_data = landmarks_sequence.reshape(1, SEQUENCE_LENGTH, expected_frame_size)
        
        # Execute prediction
        prediction = model.predict(input_data, verbose=0)[0]
        
        # Extract predicted class and confidence
        predicted_class = int(np.argmax(prediction))  # Index of class with maximum probability
        confidence = float(prediction[predicted_class])  # Confidence value
        
        # Response with classification results
        return jsonify({
            'exercise': predicted_class,
            'confidence': confidence,
            'class_name': CLASSES[predicted_class]
        })

    # Error handling
    except json.JSONDecodeError:
        return jsonify({
            'error': 'Invalid JSON',
            'exercise': None,
            'confidence': 0,
            'class_name': None
        }), 400
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'exercise': None,
            'confidence': 0,
            'class_name': None
        }), 500

# Automatic browser opening after server startup
def open_browser():
    time.sleep(2)  # Wait to ensure the server is ready
    webbrowser.open('http://localhost:8000/')

# APPLICATION STARTUP
if __name__ == "__main__":
    # Check existence of interface HTML file
    if not os.path.exists('static/index.html'):
        print("Warning: static/index.html not found")
        print("Make sure your HTML file is in the static directory")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True 
    browser_thread.start()
    
    # Start Flask server
    # host='0.0.0.0' allows connections from any network interface
    app.run(host='0.0.0.0', port=8000, debug=False)