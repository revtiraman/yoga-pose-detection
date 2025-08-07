import cv2
import mediapipe as mp
import os
import csv
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)

# --- CONFIGURATION ---
# Set the path to the folder containing your TRAIN and TEST folders.
# Based on your screenshot, it is named "DATASET".
DATASET_PATH = 'DATASET' # <<< THIS IS THE CORRECT PATH FOR YOUR STRUCTURE

# The name for our output CSV file
CSV_PATH = 'yoga_poses_landmarks.csv'

# --- SCRIPT ---
# Create the CSV file and write the header
num_landmarks = 33
header = ['class']
for i in range(num_landmarks):
    header += [f'x{i}', f'y{i}', f'z{i}', f'v{i}']

with open(CSV_PATH, mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(header)

print("Starting dataset processing...")

# Process both TRAIN and TEST directories
for split in ['TRAIN', 'TEST']:
    split_path = os.path.join(DATASET_PATH, split)
    if not os.path.isdir(split_path):
        print(f"Warning: '{split}' folder not found in dataset path. Skipping.")
        continue

    # Loop through each pose class folder (e.g., 'downdog', 'goddess')
    for pose_class in os.listdir(split_path):
        class_path = os.path.join(split_path, pose_class)
        
        if not os.path.isdir(class_path):
            continue

        print(f"Processing class: {pose_class} in {split} set")
        
        # Loop through each image in the class folder
        for image_name in os.listdir(class_path):
            image_path = os.path.join(class_path, image_name)
            
            try:
                image = cv2.imread(image_path)
                if image is None:
                    print(f"  - Could not read image: {image_name}")
                    continue
                
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    row = [pose_class] # First column is the class name
                    for lm in landmarks:
                        row.extend([lm.x, lm.y, lm.z, lm.visibility])
                    
                    with open(CSV_PATH, mode='a', newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow(row)

            except Exception as e:
                print(f"  - Error processing image {image_name}: {e}")

print(f"\nProcessing complete! Landmark data saved to {CSV_PATH}")