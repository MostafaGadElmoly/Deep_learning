import cv2
import mediapipe as mp
import time
import math

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

# Get video properties for saving
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20  # You can adjust FPS here

# Set up VideoWriter

snap_count = 0
snap_detected = [False, False]
last_snap_time = [0, 0]
snap_threshold = 0.04
cooldown_time = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

            dx = thumb_tip.x - middle_tip.x
            dy = thumb_tip.y - middle_tip.y
            distance = math.sqrt(dx * dx + dy * dy)

            current_time = time.time()

            if distance < snap_threshold and not snap_detected[i] and (current_time - last_snap_time[i]) > cooldown_time:
                snap_count += 1
                snap_detected[i] = True
                last_snap_time[i] = current_time
            elif distance >= snap_threshold:
                snap_detected[i] = False

    # Display snap count
    cv2.putText(frame, f"Snap Count: {snap_count}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Show frame
    cv2.imshow("Snap Counter", frame)

    # Write to video file

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
