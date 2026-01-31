import numpy as np
import librosa
import sounddevice as sd
from tensorflow.keras.models import load_model

# LOAD MODEL + CLASS NAMES
MODEL_PATH = "models/sound_model.h5"
model = load_model(MODEL_PATH)

# Load class names dynamically
class_names = np.load("class_names.npy")

# AUDIO PREPROCESSING
def preprocess_audio(audio, sr=22050):
    # Mel spectrogram
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Fix size (128x128)
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

    # Normalize
    mel_db = mel_db / 255.0

    # Convert to 3 channels
    mel_db = np.stack([mel_db, mel_db, mel_db], axis=-1)

    # Add batch dimension
    mel_db = np.expand_dims(mel_db, axis=0)

    return mel_db

# LIVE MICROPHONE PREDICTION
def predict_sound_live():
    sr = 22050
    duration = 1  # 1 second recording

    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()

    audio = recording[:, 0]

    mel = preprocess_audio(audio, sr)

    preds = model.predict(mel)[0]

    label = class_names[np.argmax(preds)]

    return label, preds


# FILE PREDICTION
def predict_from_file(path):
    audio, sr = librosa.load(path, sr=22050)

    mel = preprocess_audio(audio, sr)

    preds = model.predict(mel)[0]

    label = class_names[np.argmax(preds)]

    return label, preds
