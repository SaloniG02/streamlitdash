import librosa
import numpy as np

def extract_audio_features(path):
    y, sr = librosa.load(path, sr=22050)

    energy = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    flux = np.mean(librosa.onset.onset_strength(y=y, sr=sr))

    return energy, zcr, centroid, flux

def predict_audio_emotion(path):
    try:
        energy, zcr, centroid, flux = extract_audio_features(path)

        # -------------------------------------------
        # NEW RULES FOR FIGHT / ANGER / CHAOS
        # -------------------------------------------
        # If audio is loud + chaotic + high pitch → NEGATIVE
        if energy > 0.05 and flux > 0.8 and centroid > 1200:
            return "negative", 0.85   # treat as NEGATIVE, not HAPPY

        # If clear anger pattern (shouting)
        if energy > 0.05 and zcr > 0.1 and flux > 1.0:
            return "angry", 0.85

        # If calm but low energy → SAD
        if energy < 0.02 and zcr < 0.05:
            return "sad", 0.75

        # If soft + stable → NEUTRAL
        if energy < 0.05 and flux < 0.5:
            return "neutral", 0.60

        # Default / fallback
        return "negative", 0.65

    except Exception as e:
        return f"Error: {e}", None
