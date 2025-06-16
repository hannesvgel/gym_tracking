from flask import Flask, request, jsonify, send_from_directory
import webbrowser
import os
import json
import threading
import time
import numpy as np
from flask_cors import CORS

# Flask Configuration
app = Flask(__name__, static_folder='static')
CORS(app)  # Enable CORS for all routes

# Model imports
MODEL_AVAILABLE = False
model = None
CLASSES = ["bench_press", "bulgarian_squat", "lat_machine", "pull_up", "push_up", "split_squat"]
SEQUENCE_LENGTH = 30
KEYPOINT_DIM = 132  # 33 landmarks * 4 (x, y, z, visibility)

# Try to load the model
try:
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
    print("TensorFlow imported successfully")
    
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

# Main route to serve the application
@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# Route for classification
@app.route('/classify', methods=['POST'])
def classify():
    if not MODEL_AVAILABLE or model is None:
        return jsonify({
            'error': 'Model not available',
            'exercise': None,
            'confidence': 0,
            'class_name': None
        }), 503

    try:
        data = request.get_json()
        sequence = data['sequence']
        landmarks_sequence = np.array(sequence, dtype=np.float32)

        # Check dimensions
        if landmarks_sequence.shape[0] != SEQUENCE_LENGTH:
            return jsonify({
                'error': f'Invalid sequence length. Expected {SEQUENCE_LENGTH}, got {landmarks_sequence.shape[0]}',
                'exercise': None,
                'confidence': 0,
                'class_name': None
            }), 400

        expected_frame_size = KEYPOINT_DIM
        if landmarks_sequence.shape[1] != expected_frame_size:
            if landmarks_sequence.size == SEQUENCE_LENGTH * expected_frame_size:
                landmarks_sequence = landmarks_sequence.reshape(SEQUENCE_LENGTH, expected_frame_size)
            else:
                return jsonify({
                    'error': f'Invalid frame size. Each frame should have {expected_frame_size} values',
                    'exercise': None,
                    'confidence': 0,
                    'class_name': None
                }), 400

        # Prediction
        input_data = landmarks_sequence.reshape(1, SEQUENCE_LENGTH, expected_frame_size)
        prediction = model.predict(input_data, verbose=0)[0]
        predicted_class = int(np.argmax(prediction))
        confidence = float(prediction[predicted_class])
        
        return jsonify({
            'exercise': predicted_class,
            'confidence': confidence,
            'class_name': CLASSES[predicted_class]
        })

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

def open_browser():
    """Open browser after a short delay"""
    time.sleep(2)
    webbrowser.open('http://localhost:8000/')

if __name__ == "__main__":
    # Check if HTML file exists
    if not os.path.exists('static/index.html'):
        print("Warning: static/index.html not found")
        print("Make sure your HTML file is in the static directory")
    
    # Start browser in a separate thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Start Flask server
    app.run(host='0.0.0.0', port=8000, debug=False)