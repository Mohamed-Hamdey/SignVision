"""
SignVision Demo - Fixed Version
-------------------------------------------------------
Handles both .h5 and .keras models with robust error handling
"""

import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyttsx3
import time
from collections import deque, Counter
from threading import Thread, Lock
import os

# =====================================================
# 1️⃣ CONFIGURATION
# =====================================================
MODEL_PATH = "E:/Downloads/sign_language/SignVision/models/signvision_cnn.h5"
# Fallback if .keras doesn't exist
if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "E:/Downloads/sign_language/SignVision/models/best_signvision.h5"
    print(f"⚠️  .keras not found, using: {MODEL_PATH}")

IMG_SIZE = (224, 224)  # CHANGED to 224x224 (standard for most models)
MIN_CONFIDENCE = 0.70
STABLE_FRAMES = 6
SPEECH_COOLDOWN = 2.0
PADDING_PIXELS = 30

# =====================================================
# 2️⃣ LOAD MODEL AND CLASS LABELS
# =====================================================
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Loaded model from: {MODEL_PATH}")
    
    # Print model input shape
    input_shape = model.input_shape
    print(f"📐 Model expects input shape: {input_shape}")
    
    # Auto-detect image size from model
    if input_shape[1] is not None and input_shape[2] is not None:
        IMG_SIZE = (input_shape[1], input_shape[2])
        print(f"✅ Using image size: {IMG_SIZE}")
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    exit(1)

# Map numeric labels (0–4) to their meanings
# IMPORTANT: Order must match your training data folders
class_meanings = {
    0: 'yes',
    1: 'no',
    2: 'I love you',
    3: 'hello',
    4: 'thank you'
}

# Get number of classes from model
num_classes = model.output_shape[-1]
print(f"📋 Model has {num_classes} output classes")

if num_classes != len(class_meanings):
    print(f"⚠️  WARNING: Model has {num_classes} classes but you defined {len(class_meanings)} meanings!")
    print("    Adjusting class_meanings to match model...")
    # Keep only the classes that exist
    class_meanings = {k: v for k, v in class_meanings.items() if k < num_classes}

class_names = list(class_meanings.keys())
print(f"✅ Active classes: {class_meanings}")

# =====================================================
# 3️⃣ SPEECH ENGINE (ASYNC)
# =====================================================
speech_lock = Lock()
is_speaking = False

def _speak(text):
    """Blocking speech synthesis (runs in thread)."""
    global is_speaking
    try:
        with speech_lock:
            is_speaking = True
            engine = pyttsx3.init()
            engine.setProperty('rate', 160)
            engine.setProperty('volume', 1.0)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
            is_speaking = False
    except Exception as e:
        print(f"❌ TTS error: {e}")
        is_speaking = False

def speak_async(text):
    """Speak text in background thread."""
    global is_speaking
    if not is_speaking:
        Thread(target=_speak, args=(text,), daemon=True).start()

# =====================================================
# 4️⃣ MEDIAPIPE HAND DETECTOR
# =====================================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
    static_image_mode=False
)

# =====================================================
# 5️⃣ WEBCAM INIT
# =====================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam (index 0)")

# Set webcam properties for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

print("🎥 SignVision started! Press ESC to quit.")
print("=" * 50)

# =====================================================
# 6️⃣ RUNTIME VARIABLES
# =====================================================
pred_history = deque(maxlen=STABLE_FRAMES * 3)
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
            print("❌ Frame capture failed.")
            break

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        frame_count += 1

        # Convert to RGB for mediapipe
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        bbox = None
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]

            xmin = int(min(x_coords) * w) - PADDING_PIXELS
            xmax = int(max(x_coords) * w) + PADDING_PIXELS
            ymin = int(min(y_coords) * h) - PADDING_PIXELS
            ymax = int(max(y_coords) * h) + PADDING_PIXELS

            # Clamp to frame boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w - 1, xmax)
            ymax = min(h - 1, ymax)

            bbox = (xmin, ymin, xmax, ymax)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Crop hand region or use full frame
        if bbox and (bbox[2] - bbox[0] > 20 and bbox[3] - bbox[1] > 20):
            x1, y1, x2, y2 = bbox
            hand_img = frame[y1:y2, x1:x2]
        else:
            hand_img = frame

        # Preprocess and predict
        try:
            # Resize to model's expected size
            img = cv2.resize(hand_img, IMG_SIZE)
            img = img.astype(np.float32) / 255.0
            img = np.expand_dims(img, axis=0)

            # Predict
            pred = model.predict(img, verbose=0)[0]
            class_id = int(np.argmax(pred))
            conf = float(pred[class_id])

            # Validate class_id
            if class_id < 0 or class_id >= num_classes:
                print(f"⚠️  Invalid class_id: {class_id}")
                continue

            # Add to history
            pred_history.append((class_id, conf))

            # Stability check
            recent = list(pred_history)[-STABLE_FRAMES:]
            high_conf = [p[0] for p in recent if p[1] >= MIN_CONFIDENCE]

            if len(high_conf) >= STABLE_FRAMES // 2:
                label_counts = Counter(high_conf)
                most_common_label, count = label_counts.most_common(1)[0]
                
                if most_common_label == stable_label:
                    stable_count += 1
                else:
                    stable_label = most_common_label
                    stable_count = 1
            else:
                stable_count = max(0, stable_count - 1)

            # Speech decision
            now = time.time()
            if stable_label is not None and stable_count >= STABLE_FRAMES // 2:
                # Calculate average confidence for stable label
                stable_preds = [p[1] for p in pred_history if p[0] == stable_label]
                avg_conf = np.mean(stable_preds) if stable_preds else 0.0
                
                should_speak = (
                    avg_conf >= MIN_CONFIDENCE and
                    (stable_label != last_spoken_label or now - last_spoken_time > SPEECH_COOLDOWN)
                )
                
                if should_speak:
                    text = class_meanings.get(stable_label, f"Unknown {stable_label}")
                    print(f"🗣️  Speaking '{text}' (label={stable_label}, conf={avg_conf:.2f})")
                    speak_async(text)
                    last_spoken_label = stable_label
                    last_spoken_time = now

            # Display prediction
            display_label = stable_label if stable_label is not None else class_id
            meaning = class_meanings.get(display_label, f"Class {display_label}")
            display_preds = [p[1] for p in pred_history if p[0] == display_label]
            display_conf = float(np.mean(display_preds)) if display_preds else conf

            # Main text
            cv2.putText(frame, f"Sign: {meaning}", (10, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Confidence: {display_conf:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(frame, f"Stable: {stable_count}/{STABLE_FRAMES//2}", (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 1)

            # Show all class probabilities
            y_offset = 140
            for i in range(min(num_classes, 5)):  # Show max 5 classes
                if i in class_meanings:
                    name = class_meanings[i]
                    prob = pred[i] if i < len(pred) else 0.0
                    color = (0, 255, 0) if i == display_label else (200, 200, 200)
                    cv2.putText(frame, f"{i}: {name} ({prob:.2f})", 
                               (10, y_offset + i * 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Debug every 60 frames
            if frame_count % 60 == 0:
                print(f"🔍 Frame {frame_count}: pred={class_id}, stable={stable_label}, count={stable_count}")

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            import traceback
            traceback.print_exc()

        # Footer
        cv2.putText(frame, "Press ESC to quit", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow("SignVision - Real-time Sign to Speech", frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\n✅ SignVision closed successfully!")