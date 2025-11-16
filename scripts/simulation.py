import os
import joblib
import numpy as np
import pandas as pd
import librosa
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize, StandardScaler

# ------------------------------
# Paths to models and preprocessing objects
# ------------------------------
face_model_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\rf_face_recognition_model.pkl"
voice_model_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\voice_recognition_model.h5"
product_model_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\random_forest_productpredict.joblib"
face_scaler_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\face_scaler.pkl"
face_label_encoder_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\face_label_encoder.pkl"
feature_extractor_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\models\feature_extractor.h5"
merged_dataset_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\results\merged_df.csv"
authorized_face_embeddings_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\results\image_embeddings_scaled.csv"
authorized_voice_embeddings_path = r"C:\Users\fadhl\OneDrive\Desktop\Multimodal Data Preprocessing Assignment\results\audio_features.csv"

# ------------------------------
# Load models
# ------------------------------
print("Loading models...")
face_model = joblib.load(face_model_path)
product_model = joblib.load(product_model_path)
voice_model = load_model(voice_model_path)

face_scaler = joblib.load(face_scaler_path)
face_label_encoder = joblib.load(face_label_encoder_path)
feature_extractor = load_model(feature_extractor_path)
print("Models loaded successfully.\n")

# ------------------------------
# Load merged dataset
# ------------------------------
merged_df = pd.read_csv(merged_dataset_path)
print(f"Merged dataset loaded ({merged_df.shape[0]} rows, {merged_df.shape[1]} columns).")

# ------------------------------
# Load authorized embeddings
# ------------------------------
# Face embeddings (normalized)
authorized_face_embeddings = pd.read_csv(authorized_face_embeddings_path).select_dtypes(include=[np.number])
authorized_face_embeddings = normalize(authorized_face_embeddings.values)

# Voice embeddings (scaled with StandardScaler)
authorized_voice_embeddings = pd.read_csv(authorized_voice_embeddings_path).select_dtypes(include=[np.number])
voice_scaler = StandardScaler()
authorized_voice_embeddings_scaled = voice_scaler.fit_transform(authorized_voice_embeddings.values)

# ------------------------------
# Helper functions
# ------------------------------
def extract_face_features(img_path):
    img = keras_image.load_img(img_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor.predict(img_array, verbose=0).flatten()
    features = face_scaler.transform([features])
    features = normalize(features)  # normalize before similarity
    return features

def extract_voice_features(voice_path):
    """Extract features and apply the same scaler used for CSV embeddings."""
    y, sr = librosa.load(voice_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)
    mfccs = np.mean(librosa.feature.mfcc(y=y_trimmed, sr=sr, n_mfcc=13), axis=1)
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_trimmed, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y_trimmed))
    features = np.concatenate([mfccs, [rolloff, rms]]).reshape(1, -1)
    
    # Apply the same scaler as used for CSV
    features_scaled = voice_scaler.transform(features)
    return features_scaled

# ------------------------------
# Authorization checks (no prints)
# ------------------------------
def is_face_authorized(face_features, threshold=0.7):
    sims = cosine_similarity(face_features, authorized_face_embeddings)
    max_sim = sims.max()
    return max_sim >= threshold

def is_voice_authorized(voice_features, threshold=0.7):
    sims = cosine_similarity(voice_features, authorized_voice_embeddings_scaled)
    max_sim = sims.max()
    return max_sim >= threshold

# ------------------------------
# Simulation function
# ------------------------------
def run_simulation(customer_id, face_path, voice_path):
    print(f"\n=== SIMULATION FOR CUSTOMER ID: {customer_id} ===")

    # Face Recognition
    print("\n--- Step 1: Face Recognition ---")
    face_features = extract_face_features(face_path)
    if not is_face_authorized(face_features):
        print("Face not recognized. Access denied.")
        return False
    print("Face recognized successfully.")

    # Voice Verification
    print("\n--- Step 2: Voice Verification ---")
    voice_features = extract_voice_features(voice_path)
    if not is_voice_authorized(voice_features):
        print("Voice verification failed. Access denied.")
        return False
    print("Voice verified successfully.")

    # Product Recommendation
    print("\n--- Step 3: Product Recommendation ---")
    sample_input_features = merged_df[merged_df['customer_id_new'] == customer_id]
    if sample_input_features.empty:
        print(f"No data found for customer_id_new='{customer_id}'")
        return False
    feature_columns = [col for col in sample_input_features.columns if col != 'customer_id_new']
    sample_input_features = sample_input_features.reindex(columns=feature_columns, fill_value=0)
    product_prediction = product_model.predict(sample_input_features)[0]
    print(f"Recommended product: {product_prediction}")
    print("Transaction completed successfully!\n")
    return True

# ------------------------------
# Prompt user for inputs
# ------------------------------
customer_id = input("Enter customer ID: ").strip()
face_path = input("Enter path to face image: ").strip()
voice_path = input("Enter path to voice file: ").strip()

# ------------------------------
# Run simulation
# ------------------------------
run_simulation(customer_id, face_path, voice_path)
