import cv2
import numpy as np
import tensorflow as tf
from collections import deque, Counter
import pyttsx3
import threading
import time

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_PATH       = "gesture_model.h5"
IMG_SIZE         = 128
WINDOW_SIZE      = 20       # frames in sliding window
CONFIDENCE_THRESH = 0.70    # minimum confidence to count a vote
VOTE_THRESH      = 0.5      # fraction of window votes needed to confirm gesture
COOLDOWN_SECONDS = 3.0    # seconds before same gesture can trigger again

GESTURE_LABELS = [
    "come", "hello", "help", "no",
    "please", "sorry", "stop",
    "THANK_YOU", "water", "yes"
]

# ─────────────────────────────────────────────
# HSV SKIN MASK  (must match your data collection exactly)
# ─────────────────────────────────────────────
LOWER_SKIN = np.array([0,  20,  70],  dtype=np.uint8)
UPPER_SKIN = np.array([20, 255, 255], dtype=np.uint8)

def apply_skin_mask(frame):
    """
    Converts frame to HSV, creates skin mask,
    returns frame with non-skin pixels set to black.
    Tweak LOWER_SKIN / UPPER_SKIN if detection is off.
    """
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_SKIN, UPPER_SKIN)

    # Morphological cleanup to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask   = cv2.dilate(mask, kernel, iterations=2)
    mask   = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    masked = cv2.bitwise_and(frame, frame, mask=mask)
    return masked, mask


def preprocess_frame(frame):
    """
    Apply skin mask → resize → normalize.
    Matches exactly how training images were prepared.
    """
    masked, _ = apply_skin_mask(frame)
    resized   = cv2.resize(masked, (IMG_SIZE, IMG_SIZE))
    normalized = resized.astype(np.float32) / 255.0
    return np.expand_dims(normalized, axis=0)  # shape: (1, 128, 128, 3)


# ─────────────────────────────────────────────
# TEXT TO SPEECH  (runs in background thread)
# ─────────────────────────────────────────────
tts_lock = threading.Lock()

def speak(text):
    def _speak():
        with tts_lock:
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
    threading.Thread(target=_speak, daemon=True).start()

# ─────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────────
def main():
    print("[INFO] Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    print("[INFO] Model loaded.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        return

    # Sliding window buffer stores (predicted_label, confidence) per frame
    window         = deque(maxlen=WINDOW_SIZE)
    last_spoken    = ""
    last_spoken_time = 0.0
    current_display  = ""
    display_confidence = 0.0

    print("[INFO] Starting inference. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        display_frame = frame.copy()

        # ── 1. Preprocess & predict ──────────────────────────
        input_tensor = preprocess_frame(frame)
        predictions  = model.predict(input_tensor, verbose=0)[0]  # shape: (num_classes,)
        pred_idx     = int(np.argmax(predictions))
        confidence   = float(predictions[pred_idx])
        pred_label   = GESTURE_LABELS[pred_idx]

        # ── 2. Add to sliding window only if confident enough ─
        if confidence >= CONFIDENCE_THRESH:
            window.append(pred_label)

        # ── 3. Majority vote across window ───────────────────
        confirmed_gesture = None
        if len(window) == WINDOW_SIZE:
            vote_counts  = Counter(window)
            top_label, top_votes = vote_counts.most_common(1)[0]
            vote_fraction = top_votes / WINDOW_SIZE

            if vote_fraction >= VOTE_THRESH:
                confirmed_gesture = top_label

        # ── 4. Cooldown + speak ───────────────────────────────
        now = time.time()
        if confirmed_gesture:
            current_display    = confirmed_gesture
            display_confidence = confidence

            # Only speak if different gesture OR cooldown expired
            if (confirmed_gesture != last_spoken or
                    now - last_spoken_time > COOLDOWN_SECONDS):
                speak(confirmed_gesture.replace("_", " "))
                last_spoken      = confirmed_gesture
                last_spoken_time = now
                window.clear()  # reset window after confirmed prediction

        # ── 5. Draw UI on frame ───────────────────────────────
        h, w = display_frame.shape[:2]

        # Show masked region (small preview, top-right)
        masked_preview, _ = apply_skin_mask(frame)
        preview = cv2.resize(masked_preview, (160, 120))
        display_frame[10:130, w-170:w-10] = preview

        # Confidence bar
        bar_width = int(w * 0.4)
        bar_filled = int(bar_width * confidence)
        bar_color = (0, 255, 0) if confidence >= CONFIDENCE_THRESH else (0, 165, 255)
        cv2.rectangle(display_frame, (10, h-60), (10+bar_width, h-40), (50,50,50), -1)
        cv2.rectangle(display_frame, (10, h-60), (10+bar_filled, h-40), bar_color, -1)
        cv2.putText(display_frame, f"Conf: {confidence:.2f}",
                    (10, h-65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)

        # Current frame prediction (small, grey)
        cv2.putText(display_frame, f"Frame: {pred_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180,180,180), 2)

        # Confirmed gesture (big, green)
        if current_display:
            label_text = current_display.replace("_", " ").upper()
            cv2.putText(display_frame, label_text,
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 100), 3)

        # Window fill indicator
        fill_pct = len(window) / WINDOW_SIZE
        cv2.putText(display_frame, f"Buffer: {len(window)}/{WINDOW_SIZE}",
                    (10, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)

        # Cooldown indicator
        time_since = now - last_spoken_time
        if time_since < COOLDOWN_SECONDS:
            remaining = COOLDOWN_SECONDS - time_since
            cv2.putText(display_frame, f"Cooldown: {remaining:.1f}s",
                        (w//2 - 80, h-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,165,255), 1)

        cv2.imshow("Sign Language Recognition", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Inference stopped.")


if __name__ == "__main__":
    main()