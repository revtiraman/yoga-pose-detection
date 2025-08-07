# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# # Initialize MediaPipe solutions
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# def calculate_angle(a, b, c):
#     """
#     Calculates the angle between three points.
#     Args:
#         a: First point (e.g., shoulder).
#         b: Middle point (e.g., elbow).
#         c: End point (e.g., wrist).
#     Returns:
#         The angle in degrees.
#     """
#     a = np.array(a)  # First
#     b = np.array(b)  # Mid
#     c = np.array(c)  # End
    
#     radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
#     angle = np.abs(radians * 180.0 / np.pi)
    
#     if angle > 180.0:
#         angle = 360 - angle
        
#     return angle

# class PoseTransformer(VideoTransformerBase):
#     """
#     A class to process video frames, detect pose, and count bicep curls.
#     """
#     def __init__(self):
#         self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
#         self.counter = 0
#         self.stage = None

#     def transform(self, frame):
#         """
#         Processes a single video frame.
#         """
#         img = frame.to_ndarray(format="bgr24")

#         # Recolor image to RGB
#         image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
      
#         # Make detection
#         results = self.pose.process(image)
    
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
#         # Extract landmarks
#         try:
#             landmarks = results.pose_landmarks.landmark
            
#             # Get coordinates
#             shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
#             elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
#             wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
#             # Calculate angle
#             angle = calculate_angle(shoulder, elbow, wrist)
            
#             # Visualize angle
#             cv2.putText(image, str(round(angle, 2)), 
#                            tuple(np.multiply(elbow, [640, 480]).astype(int)), 
#                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
#                                 )
            
#             # Curl counter logic
#             if angle > 160:
#                 self.stage = "down"
#             if angle < 30 and self.stage =='down':
#                 self.stage="up"
#                 self.counter +=1
                       
#         except:
#             pass
        
#         # Render curl counter
#         # Setup status box
#         cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
#         # Rep data
#         cv2.putText(image, 'REPS', (15,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, str(self.counter), 
#                     (10,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
#         # Stage data
#         cv2.putText(image, 'STAGE', (65,12), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
#         cv2.putText(image, self.stage, 
#                     (60,60), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
#                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
#                                  )               
        
#         return image

# def main():
#     """
#     Main function to run the Streamlit app.
#     """
#     st.set_page_config(page_title="AI Bicep Curl Tracker", layout="wide")

#     st.title("AI Bicep Curl Tracker")
#     st.markdown("This application uses your webcam to track and count your bicep curls in real-time.")

#     # RTC Configuration for STUN/TURN servers
#     rtc_configuration = RTCConfiguration({
#         "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#     })

#     webrtc_streamer(
#         key="pose-detection",
#         video_transformer_factory=PoseTransformer,
#         rtc_configuration=rtc_configuration,
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#     )

#     st.sidebar.title("About")
#     st.sidebar.info(
#         "This is a demo application that uses MediaPipe for real-time pose estimation "
#         "to count bicep curls. The UI is built with Streamlit."
#     )

# if __name__ == "__main__":
#     main()
# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import time
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

# # --- MediaPipe Initialization ---
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

# # --- Main Application Class ---
# class YogaPoseTransformer(VideoTransformerBase):
#     """
#     This class processes video frames to detect and analyze yoga poses.
#     It uses MediaPipe for pose landmark detection and provides real-time feedback
#     on the correctness of the pose, counts repetitions, and tracks hold time.
#     """
#     def __init__(self, selected_pose: str):
#         """
#         Initializes the Yoga Pose Transformer.
#         Args:
#             selected_pose (str): The name of the yoga pose to be detected.
#         """
#         # --- Pose Detector Configuration ---
#         self.pose_detector = mp_pose.Pose(
#             min_detection_confidence=0.6, 
#             min_tracking_confidence=0.6
#         )
        
#         # --- State Variables ---
#         self.selected_pose = selected_pose
#         self.reps_counter = 0
#         self.pose_stage = "start"  # Stages: "start", "correct", "incorrect"
#         self.feedback_list = []    # To store multiple feedback points
        
#         # --- Timer Variables for Holding Poses ---
#         self.hold_timer_start = 0
#         self.hold_time_elapsed = 0
#         self.is_timer_running = False
#         self.POSE_HOLD_THRESHOLD = 3  # Seconds required to hold the pose for a rep

#     # --- Utility Functions ---
#     def _calculate_angle(self, a, b, c) -> float:
#         """
#         Calculates the angle between three 2D points.
#         Args:
#             a, b, c: Points as tuples or lists, where 'b' is the vertex.
#         Returns:
#             The angle in degrees, ranging from 0 to 180.
#         """
#         a = np.array(a)  # First point
#         b = np.array(b)  # Midpoint (vertex)
#         c = np.array(c)  # End point

#         # Calculate vectors from the vertex
#         vec_ba = a - b
#         vec_bc = c - b

#         # Calculate the angle using the dot product formula
#         cosine_angle = np.dot(vec_ba, vec_bc) / (np.linalg.norm(vec_ba) * np.linalg.norm(vec_bc))
#         angle_rad = np.arccos(np.clip(cosine_angle, -1.0, 1.0)) # Clip for stability
        
#         return np.degrees(angle_rad)

#     def _get_landmarks(self, landmarks, landmark_name: str) -> list:
#         """Extracts x, y coordinates for a given landmark."""
#         return [
#             landmarks[mp_pose.PoseLandmark[landmark_name].value].x,
#             landmarks[mp_pose.PoseLandmark[landmark_name].value].y,
#         ]

#     # --- Core Pose Analysis ---
#     def _analyze_pose(self, landmarks):
#         """
#         Routes to the specific analysis function for the selected pose.
#         This is the main dispatcher for pose evaluation.
#         """
#         self.feedback_list = []  # Reset feedback for the current frame
        
#         pose_map = {
#             "Tree Pose": self._analyze_tree_pose,
#             "Warrior II": self._analyze_warrior_ii_pose,
#             "Chair Pose": self._analyze_chair_pose,
#             "Triangle Pose": self._analyze_triangle_pose,
#             "Cobra Pose": self._analyze_cobra_pose,
#         }
        
#         analysis_func = pose_map.get(self.selected_pose)
#         if analysis_func:
#             return analysis_func(landmarks)
#         return False

#     # --- Detailed Pose Analysis Functions ---

#     def _analyze_tree_pose(self, landmarks) -> bool:
#         """Analyzes the Tree Pose (Vrikshasana)."""
#         # Determine which leg is the standing leg based on which ankle is lower
#         left_ankle_y = self._get_landmarks(landmarks, "LEFT_ANKLE")[1]
#         right_ankle_y = self._get_landmarks(landmarks, "RIGHT_ANKLE")[1]

#         if left_ankle_y > right_ankle_y: # Left leg is standing
#             standing_hip = self._get_landmarks(landmarks, "LEFT_HIP")
#             standing_knee = self._get_landmarks(landmarks, "LEFT_KNEE")
#             standing_ankle = self._get_landmarks(landmarks, "LEFT_ANKLE")
#             raised_foot = self._get_landmarks(landmarks, "RIGHT_FOOT_INDEX")
#         else: # Right leg is standing
#             standing_hip = self._get_landmarks(landmarks, "RIGHT_HIP")
#             standing_knee = self._get_landmarks(landmarks, "RIGHT_KNEE")
#             standing_ankle = self._get_landmarks(landmarks, "RIGHT_ANKLE")
#             raised_foot = self._get_landmarks(landmarks, "LEFT_FOOT_INDEX")

#         # --- Angle and Position Checks ---
#         # 1. Standing leg should be straight
#         standing_leg_angle = self._calculate_angle(standing_hip, standing_knee, standing_ankle)
#         is_leg_straight = standing_leg_angle > 165

#         # 2. Raised foot should be placed above the standing knee
#         is_foot_placed_correctly = raised_foot[1] < standing_knee[1]

#         # --- Provide Feedback ---
#         if not is_leg_straight:
#             self.feedback_list.append("Straighten your standing leg.")
#         if not is_foot_placed_correctly:
#             self.feedback_list.append("Lift your foot above the knee.")
            
#         return is_leg_straight and is_foot_placed_correctly

#     def _analyze_warrior_ii_pose(self, landmarks) -> bool:
#         """Analyzes the Warrior II Pose (Virabhadrasana II)."""
#         # Determine front leg based on which knee is further along the x-axis
#         left_knee_x = self._get_landmarks(landmarks, "LEFT_KNEE")[0]
#         right_knee_x = self._get_landmarks(landmarks, "RIGHT_KNEE")[0]

#         if left_knee_x < right_knee_x: # Left leg is front
#             front_hip, front_knee, front_ankle = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             back_hip, back_knee, back_ankle = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             front_shoulder, front_elbow, front_wrist = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["SHOULDER", "ELBOW", "WRIST"]]
#             back_shoulder, back_elbow, back_wrist = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["SHOULDER", "ELBOW", "WRIST"]]
#         else: # Right leg is front
#             front_hip, front_knee, front_ankle = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             back_hip, back_knee, back_ankle = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             front_shoulder, front_elbow, front_wrist = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["SHOULDER", "ELBOW", "WRIST"]]
#             back_shoulder, back_elbow, back_wrist = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["SHOULDER", "ELBOW", "WRIST"]]

#         # --- Angle Checks ---
#         front_leg_angle = self._calculate_angle(front_hip, front_knee, front_ankle)
#         back_leg_angle = self._calculate_angle(back_hip, back_knee, back_ankle)
#         front_arm_angle = self._calculate_angle(front_shoulder, front_elbow, front_wrist)
#         back_arm_angle = self._calculate_angle(back_shoulder, back_elbow, back_wrist)

#         # --- Conditions for Correct Pose ---
#         is_front_leg_bent = 85 < front_leg_angle < 115
#         is_back_leg_straight = back_leg_angle > 165
#         are_arms_straight = front_arm_angle > 165 and back_arm_angle > 165

#         # --- Provide Feedback ---
#         if not is_front_leg_bent:
#             self.feedback_list.append("Bend your front knee to ~90 degrees.")
#         if not is_back_leg_straight:
#             self.feedback_list.append("Keep your back leg straight.")
#         if not are_arms_straight:
#             self.feedback_list.append("Extend your arms fully.")
            
#         return is_front_leg_bent and is_back_leg_straight and are_arms_straight

#     def _analyze_chair_pose(self, landmarks) -> bool:
#         """Analyzes the Chair Pose (Utkatasana)."""
#         # Using left side, but logic applies to both
#         shoulder = self._get_landmarks(landmarks, "LEFT_SHOULDER")
#         hip = self._get_landmarks(landmarks, "LEFT_HIP")
#         knee = self._get_landmarks(landmarks, "LEFT_KNEE")
#         ankle = self._get_landmarks(landmarks, "LEFT_ANKLE")
#         elbow = self._get_landmarks(landmarks, "LEFT_ELBOW")
#         wrist = self._get_landmarks(landmarks, "LEFT_WRIST")

#         # --- Angle Checks ---
#         hip_angle = self._calculate_angle(shoulder, hip, knee)
#         knee_angle = self._calculate_angle(hip, knee, ankle)
#         arm_angle = self._calculate_angle(hip, shoulder, elbow)

#         # --- Conditions for Correct Pose ---
#         is_hips_low = hip_angle < 110
#         is_knees_bent = knee_angle < 110
#         are_arms_raised = arm_angle > 150

#         # --- Provide Feedback ---
#         if not is_hips_low:
#             self.feedback_list.append("Sit deeper, lower your hips.")
#         if not is_knees_bent:
#             self.feedback_list.append("Bend your knees more.")
#         if not are_arms_raised:
#             self.feedback_list.append("Raise your arms higher.")
            
#         return is_hips_low and is_knees_bent and are_arms_raised

#     def _analyze_triangle_pose(self, landmarks) -> bool:
#         """Analyzes the Triangle Pose (Trikonasana)."""
#         # Determine front leg based on x-position
#         left_ankle_x = self._get_landmarks(landmarks, "LEFT_ANKLE")[0]
#         right_ankle_x = self._get_landmarks(landmarks, "RIGHT_ANKLE")[0]

#         if left_ankle_x < right_ankle_x: # Left is front
#             front_hip, front_knee, front_ankle = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             back_hip, back_knee, back_ankle = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             top_shoulder, top_hip = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["SHOULDER", "HIP"]]
#         else: # Right is front
#             front_hip, front_knee, front_ankle = [self._get_landmarks(landmarks, f"RIGHT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             back_hip, back_knee, back_ankle = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["HIP", "KNEE", "ANKLE"]]
#             top_shoulder, top_hip = [self._get_landmarks(landmarks, f"LEFT_{p}") for p in ["SHOULDER", "HIP"]]

#         # --- Angle Checks ---
#         front_leg_angle = self._calculate_angle(front_hip, front_knee, front_ankle)
#         back_leg_angle = self._calculate_angle(back_hip, back_knee, back_ankle)
#         torso_angle = self._calculate_angle(top_shoulder, top_hip, front_hip)

#         # --- Conditions for Correct Pose ---
#         are_legs_straight = front_leg_angle > 165 and back_leg_angle > 165
#         is_torso_open = torso_angle > 150 # Check for open chest

#         # --- Provide Feedback ---
#         if not are_legs_straight:
#             self.feedback_list.append("Keep both legs straight.")
#         if not is_torso_open:
#             self.feedback_list.append("Open your chest towards the ceiling.")

#         return are_legs_straight and is_torso_open

#     def _analyze_cobra_pose(self, landmarks) -> bool:
#         """Analyzes the Cobra Pose (Bhujangasana)."""
#         # Assuming user is facing forward, lying on their stomach
#         shoulder = self._get_landmarks(landmarks, "LEFT_SHOULDER")
#         elbow = self._get_landmarks(landmarks, "LEFT_ELBOW")
#         wrist = self._get_landmarks(landmarks, "LEFT_WRIST")
#         hip = self._get_landmarks(landmarks, "LEFT_HIP")

#         # --- Angle and Position Checks ---
#         arm_angle = self._calculate_angle(shoulder, elbow, wrist)
#         is_shoulders_lifted = shoulder[1] < hip[1] # Shoulders should be vertically above hips
#         are_arms_bent = arm_angle < 160 # Arms should not be locked straight

#         # --- Provide Feedback ---
#         if not is_shoulders_lifted:
#             self.feedback_list.append("Lift your chest off the floor.")
#         if not are_arms_bent:
#             self.feedback_list.append("Keep a slight bend in your elbows.")
            
#         return is_shoulders_lifted and are_arms_bent

#     # --- Frame Transformation and Rendering ---
#     def transform(self, frame):
#         """
#         This is the main method called for each frame from the webcam.
#         """
#         image = frame.to_ndarray(format="bgr24")
        
#         # --- Image Processing ---
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_rgb.flags.writeable = False
#         results = self.pose_detector.process(image_rgb)
#         image_rgb.flags.writeable = True
#         image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

#         # --- Pose Logic ---
#         try:
#             landmarks = results.pose_landmarks.landmark
#             is_pose_correct = self._analyze_pose(landmarks)

#             if is_pose_correct:
#                 self.pose_stage = "correct"
#                 if not self.is_timer_running:
#                     self.hold_timer_start = time.time()
#                     self.is_timer_running = True
                
#                 self.hold_time_elapsed = time.time() - self.hold_timer_start
                
#                 if self.hold_time_elapsed >= self.POSE_HOLD_THRESHOLD:
#                     self.reps_counter += 1
#                     self.hold_timer_start = time.time() # Reset timer for next rep
                    
#             else:
#                 self.pose_stage = "incorrect"
#                 self.is_timer_running = False
#                 self.hold_time_elapsed = 0
#                 self.hold_timer_start = 0

#             # Draw landmarks on the image
#             mp_drawing.draw_landmarks(
#                 image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                 mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                 mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
#             )
#         except Exception:
#             # If no landmarks are detected, reset state
#             self.pose_stage = "start"
#             self.feedback_list = ["Make sure you are fully visible."]
#             self.is_timer_running = False
#             self.hold_time_elapsed = 0
#             pass

#         # --- UI Rendering on Frame ---
#         self._render_ui(image)
        
#         return image

#     def _render_ui(self, image):
#         """Renders the UI elements onto the video frame."""
#         # Status box
#         box_height = 120 if self.feedback_list else 80
#         cv2.rectangle(image, (0, 0), (450, box_height), (20, 20, 20), -1)

#         # 1. Reps Counter
#         cv2.putText(image, 'REPS', (15, 25), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
#         cv2.putText(image, str(self.reps_counter), (20, 75), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, (100, 255, 100), 3, cv2.LINE_AA)

#         # 2. Hold Timer
#         cv2.putText(image, 'HOLD TIME', (170, 25), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
#         timer_text = f"{self.hold_time_elapsed:.1f}s"
#         timer_color = (100, 255, 100) if self.is_timer_running else (200, 200, 200)
#         cv2.putText(image, timer_text, (175, 75), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 2, timer_color, 3, cv2.LINE_AA)
        
#         # 3. Feedback Section
#         if self.feedback_list:
#             y_pos = 100
#             for feedback in self.feedback_list[:2]: # Show max 2 lines of feedback
#                 cv2.putText(image, f"- {feedback}", (15, y_pos), 
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 200, 255), 2, cv2.LINE_AA)
#                 y_pos += 20
#         elif self.pose_stage == "correct":
#              cv2.putText(image, "HOLD STEADY!", (15, 100), 
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2, cv2.LINE_AA)


# # --- Streamlit UI ---
# def main():
#     """The main function to run the Streamlit application."""
#     st.set_page_config(page_title="Advanced AI Yoga Trainer", layout="wide")
    
#     st.title("🧘 Advanced AI Yoga Trainer")
#     st.markdown("""
#     Welcome to your personal AI yoga assistant! This tool uses your webcam to provide **real-time, detailed feedback** on your form.
    
#     **Instructions:**
#     1.  Select a pose from the sidebar.
#     2.  Allow webcam access.
#     3.  Position yourself so your full body is visible.
#     4.  Hold the pose correctly for **3 seconds** to score a rep!
#     """)

#     pose_list = ["Tree Pose", "Warrior II", "Chair Pose", "Triangle Pose", "Cobra Pose"]
    
#     # --- Sidebar Configuration ---
#     with st.sidebar:
#         st.header("⚙️ Pose Configuration")
#         selected_pose = st.selectbox("Select a Yoga Pose", pose_list)
#         st.info("Ensure you have good lighting and enough space around you.")

#         st.header("Pose Guide")
        
#         pose_info = {
#             "Tree Pose": ("Stand on one leg, placing the other foot on your inner thigh. Keep your standing leg straight and find your balance.", "https://i.imgur.com/k98c5i7.png"),
#             "Warrior II": ("Bend your front knee to 90 degrees, with your back leg straight. Extend arms parallel to the floor.", "https://i.imgur.com/yG2gSIs.png"),
#             "Chair Pose": ("Bend your knees and hips as if sitting in a chair. Keep your chest lifted and arms raised.", "https://i.imgur.com/sEP1HwM.png"),
#             "Triangle Pose": ("With straight legs, hinge at your hip. Reach one hand to the floor and extend the other to the sky, opening your chest.", "https://i.imgur.com/kLz2z4b.png"),
#             "Cobra Pose": ("Lie on your stomach. Lift your chest using back muscles, with a slight bend in the elbows to support.", "https://i.imgur.com/h6ex2kL.png")
#         }
        
#         description, image_url = pose_info.get(selected_pose)
        
#         st.image(image_url, caption=f"Reference for {selected_pose}", use_column_width=True)
#         st.write(description)

#     # --- WebRTC Streaming ---
#     webrtc_streamer(
#         key="yoga-pose-analysis-advanced",
#         video_transformer_factory=lambda: YogaPoseTransformer(selected_pose),
#         rtc_configuration=RTCConfiguration({
#             "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
#         }),
#         media_stream_constraints={"video": True, "audio": False},
#         async_processing=True,
#     )

# if __name__ == "__main__":
#     main()
# import cv2
# import numpy as np
# import streamlit as st
# from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
# import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils

# class PoseDetector(VideoTransformerBase):
#     def transform(self, frame):
#         image = frame.to_ndarray(format="bgr24")
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         result = pose.process(image_rgb)

#         if result.pose_landmarks:
#             mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
#         return image

# st.title("Yoga Pose Detection App")
# webrtc_streamer(key="yoga", video_transformer_factory=PoseDetector)
