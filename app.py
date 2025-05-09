from flask import Flask, render_template, request, Response, send_from_directory, jsonify
import cv2
import numpy as np
import os
from ultralytics import YOLO
import logging
import threading
from queue import Queue, Empty
import time
import uuid

# Set up logging with less verbose output for production
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
model = None
processing_queue = Queue(maxsize=5)  # Increased queue size for better mobile handling
results_queue = Queue(maxsize=5)  # Store multiple results for more reliable retrieval
processing_thread = None
stop_signal = threading.Event()
client_sessions = {}  # Track active client sessions

def load_model():
    """Load the YOLO model in a separate function for potential future optimization"""
    global model
    model_path = 'C:/Users/Mostafa/Downloads/deep=_(/best.pt'
    try:
        model = YOLO(model_path)
        # Set model parameters for faster inference
        model.conf = 0.25  # Confidence threshold
        logger.info(f"Model loaded successfully from {model_path}")
        return True
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

def process_frames():
    """Background thread function to process frames"""
    global model, processing_queue, results_queue, stop_signal, client_sessions
    
    logger.info("Processing thread started")
    
    while not stop_signal.is_set():
        try:
            # Non-blocking get with timeout
            frame_data = processing_queue.get(timeout=0.1)
            
            if not frame_data:
                continue
                
            frame, session_id, frame_id = frame_data
            
            if model is not None:
                # Run inference
                start_time = time.time()
                results = model.predict(frame, conf=0.25)[0]
                
                # Draw bounding boxes
                annotated_frame = results.plot()
                
                # Convert to JPEG with adaptive quality based on size
                frame_height, frame_width = annotated_frame.shape[:2]
                # Use higher compression for larger frames
                quality = 95 if (frame_width <= 480 or frame_height <= 480) else 80
                
                _, buffer = cv2.imencode('.jpg', annotated_frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                
                # Store in results queue with session info
                results_data = {
                    'session_id': session_id,
                    'frame_id': frame_id,
                    'data': buffer.tobytes(),
                    'timestamp': time.time(),
                    'detections': len(results.boxes)
                }
                
                # Update session info
                if session_id in client_sessions:
                    client_sessions[session_id]['last_processed'] = time.time()
                    client_sessions[session_id]['results'] = results_data
                
                # Put in results queue
                if results_queue.full():
                    try:
                        results_queue.get_nowait()  # Remove oldest result
                    except Empty:
                        pass
                
                results_queue.put(results_data)
                
                inference_time = time.time() - start_time
                logger.debug(f"Session {session_id}, Frame {frame_id}: Inference time: {inference_time:.3f}s, Detections: {len(results.boxes)}")
        
        except Empty:
            # No frame available, just continue
            pass
        except Exception as e:
            logger.error(f"Error in processing thread: {e}")
            time.sleep(0.1)  # Prevent CPU spinning on errors

def start_processing_thread():
    """Start the background processing thread if not already running"""
    global processing_thread, stop_signal
    
    if processing_thread is None or not processing_thread.is_alive():
        stop_signal.clear()
        processing_thread = threading.Thread(target=process_frames, daemon=True)
        processing_thread.start()
        logger.info("Started processing thread")

def cleanup_sessions():
    """Clean up inactive sessions"""
    global client_sessions
    current_time = time.time()
    inactive_sessions = []
    
    for session_id, session_data in client_sessions.items():
        # Remove sessions inactive for more than 2 minutes
        if current_time - session_data['last_active'] > 120:
            inactive_sessions.append(session_id)
    
    for session_id in inactive_sessions:
        del client_sessions[session_id]
        logger.info(f"Cleaned up inactive session {session_id}")

@app.route('/')
def index():
    """Serve the main page"""
    # Clean up old sessions periodically
    if len(client_sessions) > 0 and time.time() % 60 < 1:  # Check roughly every minute
        cleanup_sessions()
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)

@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize a new client session"""
    session_id = str(uuid.uuid4())
    client_sessions[session_id] = {
        'created': time.time(),
        'last_active': time.time(),
        'last_processed': None,
        'results': None
    }
    
    return jsonify({'session_id': session_id})

@app.route('/predict', methods=['POST'])
def predict():
    """Handle incoming frames from the client"""
    global processing_queue, client_sessions
    
    try:
        # Start the processing thread if not running
        start_processing_thread()
        
        # Get session info
        session_id = request.form.get('session_id', str(uuid.uuid4()))
        frame_id = request.form.get('frame_id', '0')
        
        # Update or create session
        if session_id not in client_sessions:
            client_sessions[session_id] = {
                'created': time.time(),
                'last_active': time.time(),
                'last_processed': None,
                'results': None
            }
        else:
            client_sessions[session_id]['last_active'] = time.time()
        
        # Get image from request
        file = request.files['image'].read()
        npimg = np.frombuffer(file, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            logger.error("Failed to decode image")
            return Response("Failed to decode image", status=400)
        
        # Add frame to processing queue
        if processing_queue.full():
            try:
                processing_queue.get_nowait()  # Remove oldest frame
            except Empty:
                pass
        
        processing_queue.put((frame, session_id, frame_id))
        
        # Get the processed result for this session if available
        if session_id in client_sessions and client_sessions[session_id]['results'] is not None:
            result_data = client_sessions[session_id]['results']['data']
            response = Response(result_data, mimetype='image/jpeg')
            response.headers['X-Frame-Id'] = frame_id
            response.headers['X-Detections'] = str(client_sessions[session_id]['results'].get('detections', 0))
            return response
        
        # Check results queue for this session
        found = False
        for _ in range(results_queue.qsize()):
            try:
                result = results_queue.get_nowait()
                results_queue.put(result)  # Put it back
                if result['session_id'] == session_id:
                    response = Response(result['data'], mimetype='image/jpeg')
                    response.headers['X-Frame-Id'] = frame_id
                    response.headers['X-Detections'] = str(result.get('detections', 0))
                    found = True
                    break
            except Empty:
                break
        
        if found:
            return response
            
        # If no result is available yet, return the original frame
        # with a message overlay indicating processing
        height, width = frame.shape[:2]
        font_scale = min(width, height) / 640  # Scale font based on image size
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Processing...', 
                   (int(10 * font_scale), int(30 * font_scale)), 
                   font, font_scale, (0, 255, 0), 
                   max(1, int(2 * font_scale)))
        
        # Use lower quality for first-time feedback to improve responsiveness
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        response = Response(buffer.tobytes(), mimetype='image/jpeg')
        response.headers['X-Frame-Id'] = frame_id
        return response
            
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        return Response(f"Error during processing: {str(e)}", status=500)

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    status = {
        'status': 'ok',
        'model_loaded': model is not None,
        'active_sessions': len(client_sessions),
        'queue_size': processing_queue.qsize()
    }
    return jsonify(status)

if __name__ == '__main__':
    # Load model before starting the server
    success = load_model()
    if not success:
        logger.warning("Failed to load model. App will start but predictions won't work.")
    
    # Start background processing thread
    start_processing_thread()
    
    # Configure server
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    port = int(os.environ.get('PORT', 5000))
    
    logger.info(f"Starting Flask app on port {port}, debug={debug_mode}")
    app.run(host='0.0.0.0', port=port, debug=debug_mode, threaded=True)