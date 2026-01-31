import streamlit as st
import numpy as np
import time

# Internal modules
from scripts.detect_live import predict_sound_live, predict_from_file
from scripts.sentiment import analyze_sentiment
from scripts.audio_sentiment import audio_sentiment
from scripts.audio_emotion import predict_audio_emotion   # Tone Emotion

# Load class names safely
class_names = np.load("class_names.npy", allow_pickle=True)
DANGER_CLASSES = ["glass", "gunshot", "man_shout", "women_shout", "fight"]

# Streamlit Config
st.set_page_config(page_title="Smart Home Ear AI", page_icon="🔊", layout="wide")

# UI Styling
st.markdown("""
    <style>
    .header-box {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        padding: 25px;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 30px;
    }

    .section {
        background: white;
        padding: 25px;
        border-radius: 14px;
        margin-bottom: 30px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.12);
    }

    .danger {
        background: #ffe1e1;
        padding: 15px;
        border-left: 8px solid #cc0000;
        border-radius: 8px;
        font-size: 22px;
        color: #880000;
        font-weight: bold;
    }

    .safe {
        background: #e0ffe7;
        padding: 15px;
        border-left: 8px solid #008a2e;
        border-radius: 8px;
        font-size: 22px;
        color: #006622;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


# HEADER
st.markdown("""
    <div class="header-box">
        <h1>🔊 Smart Home Ear – AI Dashboard</h1>
        <p>Real-Time Sound Alerts • Speech Sentiment • Text Sentiment • Tone Emotion</p>
    </div>
""", unsafe_allow_html=True)


# LIVE SOUND DETECTION
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("🎧 Live Sound Detection")

status_box = st.empty()
chart_box = st.empty()
log_box = st.empty()
logs = []

if st.button("▶ Start Listening", use_container_width=True):
    st.success("Listening started…")
    while True:
        label, probs = predict_sound_live()

        # Output
        if label in DANGER_CLASSES:
            status_box.markdown(f"<div class='danger'>🚨 DANGER: {label.upper()}</div>", unsafe_allow_html=True)
        else:
            status_box.markdown(f"<div class='safe'>🟢 SAFE: {label.upper()}</div>", unsafe_allow_html=True)

        # Safe probability graph
        safe_probs = {name: [probs[i]] for i, name in enumerate(class_names) if i < len(probs)}
        chart_box.bar_chart(safe_probs)

        logs.append(label)
        log_box.write(logs[-6:])
        time.sleep(1)

st.markdown("</div>", unsafe_allow_html=True)


# UPLOAD SOUND CLASSIFICATION
# -------------------------------------------------------------------------
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📁 Upload Audio for Sound Classification")

uploaded_audio = st.file_uploader(
    "Upload .wav or .mp3",
    type=["wav", "mp3"],
    key="sound_upload"
)

if uploaded_audio:
    with open("uploaded_sound.wav", "wb") as f:
        f.write(uploaded_audio.getbuffer())

    st.success("Audio uploaded!")

    if st.button("🔍 Classify Sound"):
        # CNN prediction
        label, probs = predict_from_file("uploaded_sound.wav")

        # Tone emotion override
        tone_emotion, tone_conf = predict_audio_emotion("uploaded_sound.wav")

        
        # FIGHT OVERRIDE LOGIC (main fix)
        # If CNN says baby_cry but tone is aggressive => fight
        if label == "baby_cry" and (tone_emotion in ["negative", "angry"]):
            label = "fight"

        # If tone indicates aggression => fight
        if tone_emotion in ["negative", "angry"]:
            label = "fight"
        

        # Final output
        if label in DANGER_CLASSES:
            st.markdown(f"<div class='danger'>🚨 DANGER: {label.upper()}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='safe'>🟢 SAFE: {label.upper()}</div>", unsafe_allow_html=True)

        # Safe probability display
        st.subheader("Confidence Scores:")
        for i, name in enumerate(class_names):
            if i < len(probs):
                st.write(f"{name}: {probs[i]:.4f}")
            else:
                st.write(f"{name}: N/A")

        st.subheader("Tone Emotion Override Info:")
        st.write(f"Tone: {tone_emotion.upper()} ({tone_conf:.3f})")

st.markdown("</div>", unsafe_allow_html=True)


# TEXT SENTIMENT ANALYSIS
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("📝 Text Sentiment Analysis")

text = st.text_area("Enter text:")

if st.button("Analyze Text Sentiment"):
    sentiment, conf = analyze_sentiment(text)
    st.write(f"Sentiment: **{sentiment.upper()}** ({conf:.3f})")

st.markdown("</div>", unsafe_allow_html=True)



# SPEECH SENTIMENT (Whisper)
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("🎤 Speech Sentiment (Audio → Text → Sentiment)")

speech_file = st.file_uploader(
    "Upload speech audio",
    type=["wav", "mp3"],
    key="speech_upload"
)

if speech_file:
    with open("speech.wav", "wb") as f:
        f.write(speech_file.getbuffer())

    st.success("Speech uploaded!")

    if st.button("Analyze Speech Sentiment"):
        text_out, senti, score = audio_sentiment("speech.wav")

        st.subheader("Transcription:")
        st.write(text_out)

        st.subheader("Sentiment:")
        if senti is None or score is None:
            st.error("⚠ No sentiment detected")
        else:
            st.write(f"**{senti.upper()}** ({score:.3f})")

st.markdown("</div>", unsafe_allow_html=True)


# AUDIO TONE EMOTION
st.markdown('<div class="section">', unsafe_allow_html=True)
st.header("🎙 Voice Tone Emotion (Audio Tone Analysis)")

tone_file = st.file_uploader(
    "Upload speech audio for tone emotion",
    type=["wav", "mp3"],
    key="tone_upload"
)

if tone_file:
    with open("tone.wav", "wb") as f:
        f.write(tone_file.getbuffer())

    st.success("Tone audio uploaded!")

    if st.button("Analyze Tone Emotion"):
        emotion, conf = predict_audio_emotion("tone.wav")

        st.subheader("Detected Tone Emotion:")
        st.write(f"**{emotion.upper()}** ({conf:.3f})")

        if emotion == "angry":
            st.error("😡 Angry tone detected")
        elif emotion == "sad":
            st.warning("😢 Sad tone detected")
        elif emotion == "happy":
            st.success("🙂 Happy tone detected")
        elif emotion == "negative":
            st.error("⚠ Negative tone detected")
        else:
            st.info("😐 Neutral tone detected")

st.markdown("</div>", unsafe_allow_html=True)
