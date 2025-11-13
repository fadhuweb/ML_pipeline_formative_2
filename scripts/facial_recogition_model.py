from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
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


# Load features and prepare for training
features_df = pd.read_csv(r'C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\results\image_features.csv')


features_df.head()


# Separate features (X) and labels (y)
feature_columns = [col for col in features_df.columns if col.startswith('feature_')]
X = features_df[feature_columns].values
y = features_df['label'].values



y
print("="*60)
print("DATASET INFORMATION")
print("="*60)
print(f"Feature matrix shape: {X.shape}")
print(f"  - Number of samples: {X.shape[0]}")
print(f"  - Number of features per sample: {X.shape[1]}")
print(f"\nLabels shape: {y.shape}")
print(f"Unique labels: {np.unique(y)}")
print(f"Number of classes: {len(np.unique(y))}")


# Encode labels to numbers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)



for i, label in enumerate(label_encoder.classes_):
    count = np.sum(y_encoded == i)
    print(f"  {label} → {i} ({count} samples)")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)



print(f"Training set:   {X_train.shape[0]:3d} samples ({X_train.shape[0]/len(X)*100:5.1f}%)")
print(f"Test set:       {X_test.shape[0]:3d} samples ({X_test.shape[0]/len(X)*100:5.1f}%)")
print(f"Total:          {len(X):3d} samples (100.0%)")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



xgb_model = XGBClassifier(
    n_estimators=50,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss',
    max_depth=2
)

xgb_model.fit(X_train_scaled, y_train)

print("✓ XGBoost model trained successfully")


y_train_pred = xgb_model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)

print("\n--- Training Set Evaluation ---")
print(f"Accuracy: {train_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_train, y_train_pred))



y_test_pred = xgb_model.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\n--- Testing Set Evaluation ---")
print(f"Accuracy: {test_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))


# Save trained model and preprocessing objects
models_path = r'C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models'
os.makedirs(models_path, exist_ok=True)


# Save model
model_file = os.path.join(models_path, 'face_recognition_model.pkl')
with open(model_file, 'wb') as f:
    pickle.dump(xgb_model, f)
print(f"✓ Model saved: {model_file}")

# Save scaler
scaler_file = os.path.join(models_path, 'face_scaler.pkl')
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler, f)
print(f"✓ Scaler saved: {scaler_file}")

# Save label encoder
encoder_file = os.path.join(models_path, 'face_label_encoder.pkl')
with open(encoder_file, 'wb') as f:
    pickle.dump(label_encoder, f)
print(f"✓ Label encoder saved: {encoder_file}")
