import os
import requests
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ----------------------------
# STEP 1: Download audio files from GitHub
# ----------------------------
GITHUB_API_URL = "https://api.github.com/repos/fadhuweb/ML_pipeline_formative_2/contents/data/audio"
SAVE_DIR = "audio_files"
PLOTS_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

response = requests.get(GITHUB_API_URL)
if response.status_code == 200:
    files = response.json()
    for file in files:
        if file["name"].endswith(".wav"):
            print(f"Downloading {file['name']}...")
            audio_data = requests.get(file["download_url"]).content
            file_path = os.path.join(SAVE_DIR, file["name"])
            with open(file_path, "wb") as f:
                f.write(audio_data)
else:
    raise Exception(f"Failed to fetch file list: {response.status_code}")



# ----------------------------
# STEP 2: Feature extraction helper
# ----------------------------
def extract_features(y, sr):
    """Extract MFCCs, spectral rolloff, and RMS energy"""
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    rms = np.mean(librosa.feature.rms(y=y))
    return mfccs, spectral_rolloff, rms

# ----------------------------
# STEP 3: Process audio files
# ----------------------------
audio_files = [f for f in os.listdir(SAVE_DIR) if f.endswith(".wav")]
if not audio_files:
    raise Exception("No audio files found in the folder.")

all_features = []

for audio_file in audio_files:
    file_path = os.path.join(SAVE_DIR, audio_file)
    print(f"\nAnalyzing {audio_file}...")

    # Load and trim audio
    y, sr = librosa.load(file_path, sr=None)
    y_trimmed, _ = librosa.effects.trim(y, top_db=20)

    # ---- Visualization: Waveform ----
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(y_trimmed, sr=sr)
    plt.title(f"Waveform - {audio_file}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{audio_file}_waveform.png"))
    plt.close()

    # ---- Visualization: Spectrogram ----
    S = librosa.feature.melspectrogram(y=y_trimmed, sr=sr, n_mels=128)
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_db, sr=sr, x_axis="time", y_axis="mel")
    plt.colorbar(format="%+2.0f dB")
    plt.title(f"Spectrogram - {audio_file}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{audio_file}_spectrogram.png"))
    plt.close()

    # ---- Extract features (original) ----
    mfccs, rolloff, rms = extract_features(y_trimmed, sr)
    features = {
        "filename": audio_file,
        "type": "original",
        **{f"mfcc_{i+1}": mfccs[i] for i in range(13)},
        "spectral_rolloff": rolloff,
        "rms_energy": rms
    }
    all_features.append(features)

    # ----------------------------
    # STEP 4: Data Augmentation
    # ----------------------------

    y_pitch = librosa.effects.pitch_shift(y_trimmed, sr=sr, n_steps=2)
    mfccs, rolloff, rms = extract_features(y_pitch, sr)
    features_pitch = {
        "filename": audio_file,
        "type": "pitch_shift(+2)",
        **{f"mfcc_{i+1}": mfccs[i] for i in range(13)},
        "spectral_rolloff": rolloff,
        "rms_energy": rms
    }
    all_features.append(features_pitch)
    y_stretch = librosa.effects.time_stretch(y_trimmed, rate=1.2)
    mfccs, rolloff, rms = extract_features(y_stretch, sr)
    features_stretch = {
        "filename": audio_file,
        "type": "time_stretch(1.2x)",
        **{f"mfcc_{i+1}": mfccs[i] for i in range(13)},
        "spectral_rolloff": rolloff,
        "rms_energy": rms
    }
    all_features.append(features_stretch)
    noise = np.random.normal(0, 0.005, len(y_trimmed))
    y_noisy = y_trimmed + noise
    mfccs, rolloff, rms = extract_features(y_noisy, sr)
    features_noise = {
        "filename": audio_file,
        "type": "noisy",
        **{f"mfcc_{i+1}": mfccs[i] for i in range(13)},
        "spectral_rolloff": rolloff,
        "rms_energy": rms
    }
    all_features.append(features_noise)

# ----------------------------
# STEP 5: Save all features to CSV
# ----------------------------
df = pd.DataFrame(all_features)
csv_path = "audio_features.csv"
df.to_csv(csv_path, index=False)
print(f"Feature extraction complete. Saved to '{csv_path}'")
print(f"Waveform and spectrogram plots saved in '{PLOTS_DIR}' folder.")