from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
import base64
from tensorflow.keras.models import load_model
import dlib
from imutils import face_utils

app = Flask(__name__)

# Load model and resources
emotion_model = load_model('video.h5')
face_detect = dlib.get_frontal_face_detector()
predictor_landmarks = dlib.shape_predictor('face_landmarks.dat')

def detect_iris_position(eye_region):
    try:
        gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
        gray_eye = cv2.medianBlur(gray_eye, 5)
        thresholded_eye = cv2.adaptiveThreshold(gray_eye, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                if cX < eye_region.shape[1] / 3:
                    return "Left"
                elif cX > 2 * eye_region.shape[1] / 3:
                    return "Right"
                else:
                    return "Center"
    except Exception as e:
        print(f"Error in detect_iris_position: {e}")
    return "Unknown"

def encode_image_to_base64(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        data = request.get_json()
        image_data = base64.b64decode(data['image'].split(',')[1])
        np_img = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = face_detect(gray, 1)
        num_faces = len(rects)
        attentive = 'no'
        emotion_label = "Unknown"
        emotion_confidences = {}
        gaze_direction = "Unknown"
        face_detected = False

        for rect in rects:
            face_detected = True
            shape = predictor_landmarks(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Draw landmarks on the frame
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Save the frame with landmarks
            cv2.imwrite('static/images/landmark.png', frame)  # Save the image here

            x, y, w, h = face_utils.rect_to_bb(rect)
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48)).reshape(1, 48, 48, 1) / 255.0
            prediction = emotion_model.predict(face_resized)[0]
            emotion_label = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"][np.argmax(prediction)]
            emotion_confidences = {emotion: round(float(prob), 2) for emotion, prob in zip(
                ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"], prediction)}
            
            # Eye extraction and gaze direction
            left_eye_points = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
            right_eye_points = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
            
            left_eye_box = cv2.boundingRect(np.array([left_eye_points]))
            right_eye_box = cv2.boundingRect(np.array([right_eye_points]))

            left_eye_img = frame[left_eye_box[1]:left_eye_box[1]+left_eye_box[3], left_eye_box[0]:left_eye_box[0]+left_eye_box[2]]
            right_eye_img = frame[right_eye_box[1]:right_eye_box[1]+right_eye_box[3], right_eye_box[0]:right_eye_box[0]+right_eye_box[2]]

            left_gaze = detect_iris_position(left_eye_img) if left_eye_img.size else "Unknown"
            right_gaze = detect_iris_position(right_eye_img) if right_eye_img.size else "Unknown"
            
            if left_gaze == "Center" and right_gaze == "Center":
                gaze_direction = "Center"
            elif left_gaze == "Left" or right_gaze == "Left":
                gaze_direction = "Left"
            elif left_gaze == "Right" or right_gaze == "Right":
                gaze_direction = "Right"

            attentive = "Yes" if gaze_direction == "Center" else "No"

        # Encode the frame with landmarks as base64
        frame_with_landmarks = encode_image_to_base64(frame)
        # Save the frame with landmarks
        cv2.imwrite('static/images/landmark.png', frame)  # Save the image here

        response = {
            "emotion": emotion_label,
            "emotion_confidences": emotion_confidences,
            "num_faces": num_faces,
            "gaze_direction": gaze_direction,
            "attentive": attentive,
            "frame_with_landmarks": f"data:image/jpeg;base64,{frame_with_landmarks}"
        }

        return jsonify(response)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({"error": "Error processing frame."}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

