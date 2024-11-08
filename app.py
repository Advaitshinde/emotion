from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import dlib
from imutils import face_utils
from scipy.spatial import distance
import base64
import h5py
import json

# Initialize Flask app
app = Flask(__name__)

# Global variables for Dlib and Keras model
shape_x, shape_y = 48, 48
model_path = 'video.h5'
face_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("face_landmarks.dat")
model = None

def modify_h5_model(file_path):
    with h5py.File(file_path, 'r+') as f:
        config = json.loads(f.attrs['model_config'])
        for layer in config['config']['layers']:
            if 'kernel_initializer' in layer['config']:
                layer['config']['kernel_initializer']['config'].pop('dtype', None)
            if 'bias_initializer' in layer['config']:
                layer['config']['bias_initializer']['config'].pop('dtype', None)
            if layer['class_name'] == 'BatchNormalization':
                for initializer in ['beta_initializer', 'gamma_initializer', 'moving_mean_initializer', 'moving_variance_initializer']:
                    if initializer in layer['config']:
                        layer['config'][initializer]['config'].pop('dtype', None)
        f.attrs['model_config'] = json.dumps(config).encode('utf-8')

# Load the emotion model    
def load_emotion_model():
    global model
    modify_h5_model(model_path)
    model = load_model(model_path)

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to process frames and add annotations
def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = face_detector(gray, 1)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    
    emotion_label = "No Face Detected"
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        (x, y, w, h) = face_utils.rect_to_bb(rect)
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (shape_x, shape_y)).astype(np.float32) / 255.0
        face = np.reshape(face, (1, shape_x, shape_y, 1))

        prediction = model.predict(face)
        prediction_result = np.argmax(prediction)
        emotion_label = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][prediction_result]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return emotion_label

# Route to handle frame processing from the website
@app.route('/process_frame', methods=['POST'])
def process_frame_route():
    data = request.json['image']
    image_data = base64.b64decode(data.split(",")[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    # Process the frame and get the emotion label
    emotion = process_frame(frame)
    return jsonify({"emotion": emotion})

# Home route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    load_emotion_model()
    app.run(host="0.0.0.0", port=5000)
