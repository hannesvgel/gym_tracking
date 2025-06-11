#server.py
# #!/usr/bin/env python3
import http.server
import socketserver
import webbrowser
import os
import json
import threading
import time

# Importazioni per il modello
try:
    import numpy as np
    from tensorflow.keras.models import load_model
    MODEL_AVAILABLE = True
    print("TensorFlow imported successfully")
except ImportError as e:
    print(f"Warning: {e}")
    print("Model classification will not be available")
    MODEL_AVAILABLE = False

# Configurazione
PORT = 8000
DIRECTORY = "."

# Configurazione modello
CLASSES = ["bench_press", "bulgarian_squat", "lat_machine", "pull_up", "push_up", "split_squat"]
SEQUENCE_LENGTH = 30
KEYPOINT_DIM = 132  # 33 landmarks * 4 (x, y, z, visibility)

# Carica il modello LSTM solo se disponibile
model = None
if MODEL_AVAILABLE:
    try:
        if os.path.exists('skeleton_lstm_multiclass6.h5'):
            model = load_model('skeleton_lstm_multiclass6.h5')
            print("Model loaded successfully")
            print(f"Model input shape: {model.input_shape}")
        else:
            print("Model file 'skeleton_lstm_multiclass6.h5' not found")
            MODEL_AVAILABLE = False
    except Exception as e:
        print(f"Error loading model: {e}")
        MODEL_AVAILABLE = False

class ExerciseHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)
    
    def end_headers(self):
        # Aggiungi headers CORS
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()
    
    def do_OPTIONS(self):
        # Gestisci richieste preflight CORS
        self.send_response(200)
        self.end_headers()
    
    def do_POST(self):
        if self.path == '/classify':
            self.handle_classification()
        else:
            self.send_response(404)
            self.end_headers()
    
    def handle_classification(self):
        if not MODEL_AVAILABLE or model is None:
            self.send_response(503)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': 'Model not available', 'exercise': None, 'confidence': 0, 'class_name': None}
            self.wfile.write(json.dumps(response).encode())
            return

        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)

        try:
            data = json.loads(post_data.decode('utf-8'))
            sequence = data['sequence']

            # Converti in numpy array
            landmarks_sequence = np.array(sequence, dtype=np.float32)

            # Verifica dimensioni
            if landmarks_sequence.shape[0] != SEQUENCE_LENGTH:
                self.send_response(400)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {
                    'error': f'Invalid sequence length. Expected {SEQUENCE_LENGTH}, got {landmarks_sequence.shape[0]}',
                    'exercise': None,
                    'confidence': 0,
                    'class_name': None
                }
                self.wfile.write(json.dumps(response).encode())
                return

            expected_frame_size = KEYPOINT_DIM
            if landmarks_sequence.shape[1] != expected_frame_size:
                if landmarks_sequence.size == SEQUENCE_LENGTH * expected_frame_size:
                    landmarks_sequence = landmarks_sequence.reshape(SEQUENCE_LENGTH, expected_frame_size)
                else:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {
                        'error': f'Invalid frame size. Each frame should have {expected_frame_size} values',
                        'exercise': None,
                        'confidence': 0,
                        'class_name': None
                    }
                    self.wfile.write(json.dumps(response).encode())
                    return

            # Predizione
            input_data = landmarks_sequence.reshape(1, SEQUENCE_LENGTH, expected_frame_size)
            prediction = model.predict(input_data, verbose=0)
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(prediction[0][predicted_class])

            response = {
                'exercise': predicted_class,
                'confidence': confidence,
                'class_name': CLASSES[predicted_class]
            }

            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        except json.JSONDecodeError as e:
            self.send_response(400)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': 'Invalid JSON', 'exercise': None, 'confidence': 0, 'class_name': None}
            self.wfile.write(json.dumps(response).encode())

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': str(e), 'exercise': None, 'confidence': 0, 'class_name': None}
            self.wfile.write(json.dumps(response).encode())

def start_server():
    """Avvia il server HTTP"""
    with socketserver.TCPServer(("", PORT), ExerciseHandler) as httpd:
        print(f"Server running at http://localhost:{PORT}/")
        print(f"Model available: {MODEL_AVAILABLE}")
        if MODEL_AVAILABLE:
            print(f"Supported exercises: {', '.join(CLASSES)}")
        print("Press Ctrl+C to stop the server")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped")
            httpd.shutdown()

def open_browser():
    """Apri il browser dopo un breve delay"""
    time.sleep(2)
    webbrowser.open(f'http://localhost:{PORT}/')

if __name__ == "__main__":
    # Verifica che il file HTML esista
    html_file = "index.html"  # O il nome del tuo file HTML
    if not os.path.exists(html_file):
        print(f"Warning: {html_file} not found in current directory")
        print("Make sure your HTML file is in the same directory as this script")
    
    # Avvia il browser in un thread separato
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Avvia il server
    start_server()