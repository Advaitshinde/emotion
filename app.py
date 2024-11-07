from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import dlib
from imutils import face_utils
from scipy.spatial import distance
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

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion_label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame

# Video stream generator
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = process_frame(frame)
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

# Route to display video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Home route
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    load_emotion_model()
    app.run(debug=True)
