import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
from collections import deque

# Load trained model
model = tf.keras.models.load_model("E:/Downloads/sign_language/SignVision/models/signvision_cnn.h5")

# Define class labels
class_names = ['A', 'B', 'C', 'D', 'E']  # replace with your actual labels

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)
engine.setProperty('volume', 1.0)

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Initialize webcam
cap = cv2.VideoCapture(0)
IMG_SIZE = (224, 224)
pred_history = deque(maxlen=10)
last_spoken = None
last_time = time.time()

def speak_word(word):
    """Speak a word if it hasn't been spoken recently."""
    global last_spoken, last_time
    if word != last_spoken or time.time() - last_time > 3:
        engine.say(word)
        engine.runAndWait()
        last_spoken = word
        last_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    bbox = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords) * w), int(max(x_coords) * w)
            ymin, ymax = int(min(y_coords) * h), int(max(y_coords) * h)
            bbox = (xmin, ymin, xmax, ymax)

    if bbox:
        x1, y1, x2, y2 = bbox
        hand_img = frame[y1:y2, x1:x2]
    else:
        hand_img = frame

    img = cv2.resize(hand_img, IMG_SIZE)
    img = np.expand_dims(img / 255.0, axis=0)

    pred = model.predict(img, verbose=0)
    class_id = np.argmax(pred)
    conf = np.max(pred)
    label = class_names[class_id]

    pred_history.append(label)
    smoothed_pred = max(set(pred_history), key=pred_history.count)

    # Speak only if confident and stable
    if conf > 0.9:
        speak_word(smoothed_pred)

    cv2.putText(frame, f"{smoothed_pred} ({conf:.2f})", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("SignVision - Real-time Sign to Speech", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
