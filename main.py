import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load the trained model for character recognition
model = tf.keras.models.load_model('/cnn_model_weights.h5')

# Set up Mediapipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Open the camera
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width of the frame
cap.set(4, 480)  # Set height of the frame

# Initialize drawing parameters
drawing = False
color = (0, 255, 0)  # Initial color is green
prev_x, prev_y = 0, 0

# Create a blank canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Define the region of interest (ROI) for character recognition
roi = (100, 100, 300, 300)

# Initialize the recognized characters string
recognized_characters = ""

# Main loop
while True:
    # Read the camera frame
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe hands
    results = hands.process(frame_rgb)

    # Get hand landmarks if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract the coordinates of the index finger tip
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Convert the coordinates to pixel values
            x, y = int(index_finger_tip.x * frame.shape[1]), int(index_finger_tip.y * frame.shape[0])

            if drawing:
                # Draw a line between the current and previous finger position on the canvas
                cv2.line(canvas, (prev_x, prev_y), (x, y), color, 3)

            # Update the previous finger position
            prev_x, prev_y = x, y

    # Add a black box as the image sent to the model for recognition
    frame_with_box = frame.copy()
    cv2.rectangle(frame_with_box, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]), (0, 0, 0), -1)
    frame_with_box = cv2.add(frame_with_box, canvas)

    # Display the recognized characters on the screen
    cv2.putText(frame_with_box, recognized_characters, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('Finger Drawing', frame_with_box)

    # Keyboard input handling
    key = cv2.waitKey(1) & 0xFF

    # Change drawing color with 'r' and 'g' keys
    if key == ord('r'):
        color = (0, 0, 255)  # Set color to red
    elif key == ord('g'):
        color = (0, 255, 0)  # Set color to green

    # Start and stop drawing with 'a' and 's' keys
    if key == ord('a'):
        drawing = True
    elif key == ord('s'):
        drawing = False

    # Clear the canvas with 'c' key
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
        recognized_characters = ""  # Clear the recognized characters string

    # Recognize characters and append to the recognized characters string with 'p' key
    if key == ord('p'):
        roi_frame = frame[roi[1]:roi[1] + roi[3], roi[0]:roi[0] + roi[2]]
        roi_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        roi_frame = cv2.resize(roi_frame, (28, 28))
        roi_frame = np.expand_dims(roi_frame, axis=-1)
        roi_frame = np.expand_dims(roi_frame, axis=0)
        roi_frame = roi_frame.astype('float32') / 255.0

        # Recognize the character using the trained model
        prediction = model.predict(roi_frame)
        predicted_label = np.argmax(prediction)
        recognized_characters += str(chr(predicted_label + 81))

    # Check for the 'q' key to exit the program
    if key == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
