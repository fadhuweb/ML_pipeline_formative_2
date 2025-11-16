# ML Pipeline - Multimodal Data Preprocessing

A machine learning pipeline for audio and image feature extraction, preprocessing, and face recognition model training.

## Quick Start

### 1. Setup

```bash
# Clone repository
git clone https://github.com/fadhuweb/ML_pipeline_formative_2.git
cd ML_pipeline_formative_2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Preprocessing

Open and run the notebooks:
- `notebooks/01_audio_preprocessing.ipynb` - Extract audio features
- `notebooks/02_image_preprocessing.ipynb` - Extract image features

### 3. Train Model

Use the extracted features to train the face recognition model.

---

## Project Structure

```
ML_pipeline_formative_2/
├── src/
│   ├── preprocessing/       # Audio & image processing
│   ├── models/             # ML models
│   └── utils/              # Helper functions
├── notebooks/              # Jupyter notebooks
├── data/
│   ├── raw/               # Original files
│   └── processed/         # Extracted features
├── models/
│   └── trained/           # Saved models
├── results/               # Output files
└── tests/                 # Unit tests
```

---

## What Each Component Does

### Audio Processing
- Loads and trims audio files
- Extracts features: MFCCs, spectral rolloff, RMS energy
- Data augmentation: pitch shift, time stretch, noise addition

### Image Processing
- Loads and resizes images
- Extracts features using VGG16 (pre-trained)
- Data augmentation: brightness, contrast, blur, rotation

### Model Training
- Combines audio and image features
- Trains XGBoost classifier for face recognition
- Evaluates model performance

---

## Dependencies

- **Data**: numpy, pandas
- **Audio**: librosa
- **Images**: pillow, opencv-python
- **ML**: scikit-learn, xgboost, tensorflow
- **Visualization**: matplotlib
- **Testing**: pytest

---

## Usage

### Import Modules

```python
from src.preprocessing import AudioPreprocessor, ImagePreprocessor
from src.models import FaceRecognitionModel
```

### Process Audio

```python
from src.preprocessing import AudioPreprocessor

preprocessor = AudioPreprocessor(sr=22050)
y, sr = preprocessor.load_and_trim('audio.wav')
mfccs, rolloff, rms = preprocessor.extract_features(y, sr)
```

### Process Images

```python
from src.preprocessing import ImagePreprocessor

preprocessor = ImagePreprocessor(target_size=(224, 224))
img = preprocessor.load_image('image.jpg')
augmented = preprocessor.augment_image(img)
```

### Train Model

```python
from src.models import FaceRecognitionModel

model = FaceRecognitionModel()
model.train(X_train, y_train)
metrics = model.evaluate(X_test, y_test)
model.save('model.pkl', 'scaler.pkl', 'encoder.pkl')
```

---

## Data Flow

```
Raw Data -> Load & Preprocess -> Extract Features -> Save CSV
                                                       |
                                                  Train Model
                                                       |
                                          Evaluate & Save Results
```

---

## Testing

```bash
python -m pytest tests/
```

---

## File Paths

| Item | Path |
|------|------|
| Raw audio | `data/raw/audio/` |
| Raw images | `data/raw/images/` |
| Features | `data/processed/` |
| Trained models | `models/trained/` |
| Results | `results/` |

---

## Important Files

- **Audio notebook**: `notebooks/01_audio_preprocessing.ipynb`
- **Image notebook**: `notebooks/02_image_preprocessing.ipynb`
- **Config**: `requirements.txt`
- **Setup**: `setup.sh` (Linux/Mac) or `setup.bat` (Windows)

---

## Troubleshooting

**Missing dependencies?**
```bash
pip install -r requirements.txt
```

**Data not found?**
```bash
# Check your files are in the right location
ls data/raw/audio/
ls data/raw/images/
```

**Import errors?**
```bash
# Ensure you're in the project directory and virtual environment is activated
pwd
source venv/bin/activate
```

---

## Team

- Fadhlullah
- Makuochi
- Mugisha
- Melissa

---

## License

Part of ALU formative assignment 2.

---

**Last Updated:** November 2025
