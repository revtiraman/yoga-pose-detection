import numpy as np
import mediapipe as mp

# Initialize a MediaPipe Pose object to access landmark enums
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.
    """
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def get_pose_feedback(predicted_class, landmarks):
    """
    Analyzes the pose based on the predicted class and landmarks,
    and returns feedback and a color for display.
    """
    feedback = "Analyzing..."
    color = (0, 255, 0)  # Green for good form

    # --- WARRIOR II POSE ---
    if predicted_class == 'warrior2':
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        l_wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        arm_angle = calculate_angle(l_shoulder, l_elbow, l_wrist)

        if knee_angle > 100:
            feedback = "Bend your front knee more."
            color = (0, 0, 255)  # Red for correction
        elif knee_angle < 80:
            feedback = "Ease up on your front knee bend."
            color = (0, 0, 255)
        elif arm_angle < 160:
            feedback = "Straighten your arms."
            color = (0, 0, 255)
        else:
            feedback = "Excellent Form!"

    # --- PLANK POSE ---
    elif predicted_class == 'plank':
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
        
        if body_angle < 165:
            feedback = "Lower your hips to form a straight line."
            color = (0, 0, 255)
        elif body_angle > 178:
            feedback = "Keep your back straight, avoid arching."
            color = (0, 0, 255)
        else:
            feedback = "Great Form!"

    # --- TREE POSE ---
    elif predicted_class == 'tree':
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        standing_leg_angle = calculate_angle(l_hip, l_knee, l_ankle)
        
        if standing_leg_angle < 165:
            feedback = "Straighten your standing leg."
            color = (0, 0, 255)
        else:
            feedback = "Excellent Form!"

    # --- GODDESS POSE ---
    elif predicted_class == 'goddess':
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
        
        if knee_angle > 110:
            feedback = "Sink deeper into the pose, bend your knees more."
            color = (0, 0, 255)
        else:
            feedback = "Excellent Form!"
    
    # --- DOWNWARD DOG POSE ---
    elif predicted_class == 'downdog':
        l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        l_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        l_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        body_angle = calculate_angle(l_shoulder, l_hip, l_ankle)
        
        if body_angle < 80:
            feedback = "Push your hips up and back more."
            color = (0, 0, 255)
        elif body_angle > 110:
            feedback = "Bring your shoulders forward slightly."
            color = (0, 0, 255)
        else:
            feedback = "Excellent Form!"

    return feedback, color