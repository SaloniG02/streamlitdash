import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier

class CustomEncoderWav2vec2(EncoderClassifier):
    def classify_file(self, path):
        signal, fs = torchaudio.load(path)
        return self.classify_batch(signal)
