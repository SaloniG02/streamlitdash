from transformers import pipeline

sentiment_model = pipeline("sentiment-analysis")

def analyze_sentiment(text):
    result = sentiment_model(text)[0]

    label = result["label"]
    confidence = float(result["score"])

    # If model outputs NEUTRAL → force NEGATIVE
    if label.upper() == "NEUTRAL":
        label = "NEGATIVE"

    # Low confidence → also NEGATIVE
    if confidence < 0.65:
        label = "NEGATIVE"

    return label, confidence
