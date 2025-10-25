import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
from collections import deque, Counter
from threading import Thread, Lock

# =====================================================
# 1️⃣ CONFIGURATION
# =====================================================
MODEL_PATH = "E:/Downloads/sign_language/SignVision/models/signvision_cnn_augmented.h5"
IMG_SIZE = (224, 224)
MIN_CONFIDENCE = 0.70
BASE_STABLE_FRAMES = 6
SPEECH_COOLDOWN = 2.0
PADDING_RATIO = 0.15  # relative to hand size

# =====================================================
# 2️⃣ LOAD MODEL AND CLASS LABELS
# =====================================================
print("⚙️ Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"✅ Model loaded from: {MODEL_PATH}")

# Optional speed boost (mixed precision)
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Enable GPU memory growth (avoids freeze)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"🚀 Using GPU: {gpus[0].name}")
    except Exception as e:
        print("⚠️ GPU setup warning:", e)

# Class meanings
class_meanings = {
    0: 'yes',
    1: 'no',
    2: 'I love you',
    3: 'hello',
    4: 'thank you'
}
num_classes = model.output_shape[-1]

# Sanity check
if num_classes != len(class_meanings):
    print(f"⚠️ Model has {num_classes} outputs, adjusting meanings...")
    class_meanings = {i: class_meanings.get(i, f"class_{i}") for i in range(num_classes)}

# =====================================================
# 3️⃣ TEXT-TO-SPEECH ENGINE
# =====================================================
speech_lock = Lock()
is_speaking = False

def _speak(text):
    """Blocking TTS in thread."""
    global is_speaking
    with speech_lock:
        try:
            is_speaking = True
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"❌ TTS error: {e}")
        finally:
            is_speaking = False

def speak_async(text):
    """Non-blocking async TTS."""
    if not is_speaking:
        Thread(target=_speak, args=(text,), daemon=True).start()

# =====================================================
# 4️⃣ MEDIAPIPE HAND DETECTOR
# =====================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.6,
    static_image_mode=False
)

# =====================================================
# 5️⃣ WEBCAM SETUP
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 SignVision running (Press ESC to quit)")
print("=" * 50)

# =====================================================
# 6️⃣ STATE VARIABLES
# =====================================================
pred_history = deque(maxlen=BASE_STABLE_FRAMES * 3)
stable_label = None
stable_count = 0
last_spoken_label = None
last_spoken_time = 0.0
frame_count = 0

# =====================================================
# 7️⃣ MAIN LOOP
# =====================================================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Frame read error.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        # Detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        bbox = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords)*w), int(max(x_coords)*w)
            ymin, ymax = int(min(y_coords)*h), int(max(y_coords)*h)

            # Add relative padding
            pad_x = int((xmax - xmin) * PADDING_RATIO)
            pad_y = int((ymax - ymin) * PADDING_RATIO)
            xmin, xmax = max(0, xmin - pad_x), min(w-1, xmax + pad_x)
            ymin, ymax = max(0, ymin - pad_y), min(h-1, ymax + pad_y)

            bbox = (xmin, ymin, xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Crop hand
        hand_img = frame[ymin:ymax, xmin:xmax] if bbox else frame

        try:
            img = cv2.resize(hand_img, IMG_SIZE)
            img = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)

            pred = model.predict(img, verbose=0)[0]
            class_id = int(np.argmax(pred))
            conf = float(pred[class_id])
            pred_history.append((class_id, conf))

            # Weighted stability check
            weights = np.linspace(0.2, 1.0, len(pred_history))
            weighted_preds = {}
            for (cid, c), w_ in zip(pred_history, weights):
                weighted_preds[cid] = weighted_preds.get(cid, 0) + w_ * c

            best_label = max(weighted_preds, key=weighted_preds.get)
            avg_conf = weighted_preds[best_label] / np.sum(weights)

            if avg_conf >= MIN_CONFIDENCE:
                if best_label == stable_label:
                    stable_count += 1
                else:
                    stable_label = best_label
                    stable_count = 1
            else:
                stable_count = max(0, stable_count - 1)

            # Adaptive stability threshold
            dynamic_stable = max(3, BASE_STABLE_FRAMES - int(conf * 3))

            # Speech decision
            now = time.time()
            if (
                stable_label is not None and
                stable_count >= dynamic_stable and
                (stable_label != last_spoken_label or now - last_spoken_time > SPEECH_COOLDOWN)
            ):
                text = class_meanings.get(stable_label, f"class_{stable_label}")
                print(f"🗣️ Speaking: {text} (conf={avg_conf:.2f})")
                speak_async(text)
                last_spoken_label = stable_label
                last_spoken_time = now

            # Display
            meaning = class_meanings.get(stable_label, f"class_{stable_label}")
            cv2.putText(frame, f"Sign: {meaning}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Confidence: {avg_conf:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(frame, f"Stable: {stable_count}/{dynamic_stable}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,200,0), 1)

            y_offset = 140
            for i in range(num_classes):
                prob = pred[i]
                color = (0,255,0) if i == stable_label else (180,180,180)
                cv2.putText(frame, f"{i}: {class_meanings[i]} ({prob:.2f})", (10, y_offset + i*25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        except Exception as e:
            print(f"❌ Prediction error: {e}")

        cv2.putText(frame, "Press ESC to quit", (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
        cv2.imshow("SignVision - Real-time Sign to Speech", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✅ SignVision closed successfully!")
