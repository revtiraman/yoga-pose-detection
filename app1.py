import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
# Import our new logic module
from pose_logic import get_pose_feedback

# --- LOAD THE TRAINED MODEL ---
try:
    with open('yoga_pose_classifier.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file 'yoga_pose_classifier.pkl' not found. Please run train_model.py to generate it.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()


# --- MEDIAPIPE SETUP ---
mp_pose = mp.solutions.pose
pose = mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# --- STREAMLIT APP INTERFACE ---
st.title("🧘 AI Yoga Pose Corrector")
st.markdown("This app uses your webcam to analyze your yoga pose and provide real-time feedback.")

st.sidebar.title("Controls")
run = st.sidebar.checkbox('Start Webcam')

# --- MAIN APP LOGIC ---
FRAME_WINDOW = st.image([])
feedback_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
    else:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                st.write("Webcam feed ended.")
                break

            # Flip the frame horizontally for a more intuitive, mirror-like view
            frame = cv2.flip(frame, 1)
            # Convert the BGR image to RGB for processing
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            
            # Process with MediaPipe
            results = pose.process(image)
            
            # Draw the pose annotation on the image.
            # To draw on the image, we need to make it writeable again.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    
                    # Feature Extraction for Model Prediction
                    row = []
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    # Make Prediction
                    X = pd.DataFrame([row])
                    predicted_class = model.predict(X)[0]
                    prediction_prob = model.predict_proba(X)[0]
                    confidence = round(prediction_prob[np.argmax(prediction_prob)], 2)
                    
                    # Display Prediction and Confidence on the image
                    cv2.putText(image, f"POSE: {predicted_class.replace('_', ' ').title()}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(image, f"CONF: {confidence}", (10, 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                    # Get feedback if confidence is high
                    if confidence > 0.8:
                        feedback, color = get_pose_feedback(predicted_class, landmarks)
                    else:
                        feedback = "Analyzing..."
                        color = (0, 255, 0) # Green

                    # Display Feedback on the Streamlit page
                    feedback_placeholder.markdown(f"<h2 style='text-align: center; color: {'red' if color == (0,0,255) else 'green'};'>{feedback}</h2>", unsafe_allow_html=True)

                    # Draw landmarks on the image
                    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            
            except Exception as e:
                # You can uncomment the line below to help debug errors if they occur
                # st.write(f"An error occurred: {e}")
                pass

            # Display the final image in the Streamlit app
            FRAME_WINDOW.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
        cap.release()
        st.write("Webcam stopped.")
else:
    st.write("Check the 'Start Webcam' box in the sidebar to begin pose detection.")

