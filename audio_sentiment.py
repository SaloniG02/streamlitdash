import whisper
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import torch

# Sentiment model
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)

labels = ["NEGATIVE", "NEUTRAL", "POSITIVE"]

def load_whisper_model():
    return whisper.load_model("base")

def analyze_sentiment_text(text):
    encoded = tokenizer(text, return_tensors="pt")
    output = sentiment_model(**encoded)

    scores = torch.softmax(output.logits, dim=1).detach().numpy()[0]
    idx = np.argmax(scores)
    sentiment = labels[idx]
    confidence = float(scores[idx])

    # Neutral threshold
    if confidence < 0.65:
        sentiment = "NEUTRAL"

    return sentiment, confidence

def audio_sentiment(audio_path):
    try:
        whisper_model = load_whisper_model()

        result = whisper_model.transcribe(audio_path)
        text = result["text"].strip()

        if not text:
            return "No speech detected", None, None

        sentiment, confidence = analyze_sentiment_text(text)
        return text, sentiment, confidence

    except Exception as e:
        return f"Error: {e}", None, None
