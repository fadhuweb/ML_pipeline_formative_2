from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
from PIL import Image
import cv2
from glob import glob
from PIL import ImageEnhance, ImageFilter
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D


# Load features
features_df = pd.read_csv(
    r'C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\results\image_features.csv'
)

# Separate features and labels
feature_columns = [col for col in features_df.columns if col.startswith('feature_')]
X = features_df[feature_columns].values
y = features_df['label'].values

print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Feature matrix shape: {X.shape}")
print(f"  - Number of samples: {X.shape[0]}")
print(f"  - Number of features per sample: {X.shape[1]}")
print(f"\nLabels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")
print(f"Number of classes: {len(np.unique(y))}")

# Label encoding
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

for i, label in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"  {label}: {i} ({count} samples)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)

print(f"Training set:   {X_train.shape[0]:3d} samples")
print(f"Test set:       {X_test.shape[0]:3d} samples")
print(f"Total:          {len(X):3d} samples")

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight='balanced'
)

rf_model.fit(X_train_scaled, y_train)
print("Random Forest model trained successfully")

# --- Training Evaluation ---
y_train_pred = rf_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

print("\n--- Training Set Evaluation ---")
print(f"Accuracy: {train_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))

# --- Testing Evaluation ---
y_test_pred = rf_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n--- Testing Set Evaluation ---")
print(f"Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

models_path = r'C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models'
os.makedirs(models_path, exist_ok=True)

# Save Random Forest model
model_file = os.path.join(models_path, 'rf_face_recognition_model.pkl')
with open(model_file, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"Random Forest model saved: {model_file}")

# Save scaler
scaler_file = os.path.join(models_path, 'face_scaler.pkl')
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Scaler saved: {scaler_file}")

# Save label encoder
encoder_file = os.path.join(models_path, 'face_label_encoder.pkl')
with open(encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"Label encoder saved: {encoder_file}")
