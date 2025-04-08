import time
import math
import numpy as np
import cv2
import mediapipe as mp

import mujoco
import glfw

#------------------------------------------------------------------------------
# 1) INITIALIZE MUJOCO (LOAD MODEL, CREATE DATA & VIEWER CONTEXT)
#------------------------------------------------------------------------------

XML_MODEL_PATH = "my_robot.xml"  # <-- Replace with your actual MuJoCo XML file

def keyboard_callback(window, key, scancode, action, mods):
    """
    Allows closing the MuJoCo window with ESC key.
    """
    if action == glfw.PRESS and key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

def main():
    # Load model & create data
    try:
        model = mujoco.MjModel.from_xml_path(XML_MODEL_PATH)
    except Exception as e:
        print(f"Could not load model from {XML_MODEL_PATH}: {e}")
        return
    data = mujoco.MjData(model)

    # Create default camera, option structs, scene, and context
    cam = mujoco.MjvCamera()
    opt = mujoco.MjvOption()
    scn = mujoco.MjvScene(model, maxgeom=1000)
    con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)

    # Initialize GLFW
    if not glfw.init():
        print("Could not initialize GLFW.")
        return

    # Create window
    width, height = 1200, 900
    window = glfw.create_window(width, height, "MuJoCo Finger 3D Control", None, None)
    if not window:
        print("Could not create GLFW window.")
        glfw.terminate()
        return

    glfw.make_context_current(window)
    glfw.set_key_callback(window, keyboard_callback)

    #------------------------------------------------------------------------------
    # 2) INITIALIZE MEDIAPIPE HANDS + OPENCV CAPTURE
    #------------------------------------------------------------------------------

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        glfw.terminate()
        return

    # We'll track the index fingertip (you can pick another finger if you like).
    # By default:
    #   x, y are normalized to [0..1] across the image,
    #   z is a relative depth measure (negative = behind palm).
    finger_x = 0.5
    finger_y = 0.5
    finger_z = 0.0  # Start “flat”

    #------------------------------------------------------------------------------
    # 3) MAIN LOOP: READ CAMERA, PROCESS HAND, STEP SIM, RENDER
    #------------------------------------------------------------------------------

    while not glfw.window_should_close(window):
        # A) Process events
        glfw.poll_events()

        # B) Read from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame from webcam.")
            break

        # Flip for selfie view, then convert to RGB for MediaPipe
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(rgb_frame)

        # C) Extract finger 3D position
        if results.multi_hand_landmarks:
            # We'll consider only the first hand found
            hand_landmarks = results.multi_hand_landmarks[0]

            # Choose the index fingertip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            finger_x = index_tip.x  # in [0..1]
            finger_y = index_tip.y  # in [0..1], note: top-left corner is (0,0), so be mindful of direction
            finger_z = index_tip.z  # relative depth

            # Optionally, draw the landmarks on the frame
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # D) Map (finger_x, finger_y, finger_z) → Robot controls
        # Example: 3 controls in data.ctrl:
        #   data.ctrl[0] = X axis mapped to [-1..1]
        #   data.ctrl[1] = Y axis mapped to [-1..1]
        #   data.ctrl[2] = Z axis mapped to [-1..1]
        #
        # In practice, you should clamp or scale these properly for your robot’s joint or end-effector range.

        # Convert from [0..1] to [-1..1] for x and y
        ctrl_x = (finger_x - 0.5) * 2.0
        ctrl_y = (finger_y - 0.5) * 2.0

        # The z from MediaPipe is often negative for a finger in front of the palm.
        # You might want to invert or offset it. For example:
        ctrl_z = -index_tip.z  # just invert so bigger z means “further from camera”
        # or you can do any scale/offset you like: ctrl_z = -(finger_z * 2.0 + 0.5)

        # Safety clamp in [-1..1]
        ctrl_x = max(-1.0, min(1.0, ctrl_x))
        ctrl_y = max(-1.0, min(1.0, ctrl_y))
        ctrl_z = max(-1.0, min(1.0, ctrl_z))

        # Write to MuJoCo controls
        data.ctrl[0] = ctrl_x
        data.ctrl[1] = ctrl_y
        if len(data.ctrl) > 2:
            data.ctrl[2] = ctrl_z

        # E) Step the simulation
        mujoco.mj_step(model, data)

        # F) Render the scene in the MuJoCo window
        mujoco.mjv_updateScene(
            model,
            data,
            opt,
            None,
            cam,
            mujoco.mjtCatBit.mjCAT_ALL.value,
            scn
        )
        viewport = mujoco.MjrRect(0, 0, width, height)
        mujoco.mjr_render(viewport, scn, con)

        # G) Show the webcam feed with overlay
        cv2.putText(frame, f"Finger X: {finger_x:.2f}, Y: {finger_y:.2f}, Z: {finger_z:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Webcam Finger Tracking (3D)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # H) Swap buffers
        glfw.swap_buffers(window)

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    glfw.terminate()

if __name__ == "__main__":
    main()
