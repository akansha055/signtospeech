import cv2
import os
gestures = [
    "HELLO", "YES", "NO", "STOP", "COME",
    "PLEASE", "THANK_YOU", "SORRY", "WATER", "HELP"
]

dataset_path = "Dataset"
frames_per_sample = 30
samples_per_gesture = 20
for gesture in gestures:
    folder_path = os.path.join(dataset_path, gesture)
    if not os.path.exists(folder_path):
        print(f"Missing folder: {folder_path}")
        print("Please create it before running the script.")
        exit()
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Data Collection", 1000, 700)

gesture_index = 0
sample_count = 0
frame_count = 0
recording = False

print("Press SPACE to start recording")
print("Press N for next gesture")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.resize(frame, (960, 540))
    frame = cv2.flip(frame, 1)

    gesture_name = gestures[gesture_index]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (300, 110), (0, 0, 0), -1)
    alpha = 0.5
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    cv2.putText(frame, f"Gesture : {gesture_name}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, f"Sample  : {sample_count}/{samples_per_gesture}", (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

    if recording:
        cv2.putText(frame, f"Frame   : {frame_count}/{frames_per_sample}", (15, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imshow("Data Collection", frame)
    key = cv2.waitKey(1)
    if key == 32 and not recording and sample_count < samples_per_gesture:
        print("Recording started...")
        recording = True
        frame_count = 0
    if recording:
        save_path = os.path.join(
            dataset_path,
            gesture_name,
            f"{sample_count}_{frame_count}.jpg"
        )
        cv2.imwrite(save_path, frame)
        frame_count += 1

        if frame_count == frames_per_sample:
            recording = False
            sample_count += 1
            print(f"Saved sample {sample_count} for {gesture_name}")

    if key == ord('n'):
        gesture_index += 1
        sample_count = 0

        if gesture_index == len(gestures):
            print("All gestures completed!")
            break

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()