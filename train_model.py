import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# 1. Load the dataset you just created
df = pd.read_csv('yoga_poses_landmarks.csv')

# 2. Separate Features (X) and Labels (y)
# X is all the coordinate data
X = df.drop('class', axis=1)
# y is the name of the pose
y = df['class']

# 3. Split the data into training and testing sets
# This ensures we can test our model on data it has never seen before.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Initialize and Train the RandomForest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
print("Training the model...")
model.fit(X_train, y_train)
print("Training complete!")

# 5. Evaluate the model on the test set
print("\nEvaluating the model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 6. Save the trained model to a file
with open('yoga_pose_classifier.pkl', 'wb') as f:
    pickle.dump(model, f)

print(f"\nModel saved to yoga_pose_classifier.pkl")