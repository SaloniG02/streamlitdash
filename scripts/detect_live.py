import sounddevice as sd
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
MODEL_PATH = "models/sound_model.h5"
model = load_model(MODEL_PATH)

# Classes you trained the model on
class_names = ["glass", "other"]

def record_audio(duration=1, sr=22050):
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1)
    sd.wait()
    return audio.flatten()

def audio_to_melspectrogram(audio, sr=22050):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Resize spectrogram to (128, 128)
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=0)
    mel_db = librosa.util.fix_length(mel_db, size=128, axis=1)

    # Normalize to 0–1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    return mel_db

def predict_sound_live():
    audio = record_audio()

    mel = audio_to_melspectrogram(audio)

    # Convert to 3 channels (RGB like model expects)
    mel = np.stack([mel, mel, mel], axis=-1)
    mel = np.expand_dims(mel, axis=0)

    preds = model.predict(mel)[0]
    label = class_names[np.argmax(preds)]

    return label, preds.tolist()

# ------------------------------------------------------
# NEW FUNCTION: Predict from uploaded audio file
# ------------------------------------------------------
def predict_from_file(file_path, sr=22050):
    audio, _ = librosa.load(file_path, sr=sr)

    mel = audio_to_melspectrogram(audio, sr)

    mel = np.stack([mel, mel, mel], axis=-1)
    mel = np.expand_dims(mel, axis=0)

    preds = model.predict(mel)[0]
    label = class_names[np.argmax(preds)]

    return label, preds.tolist()
