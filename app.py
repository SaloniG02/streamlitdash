import streamlit as st
import numpy as np
import time
from scripts.detect_live import predict_sound_live, predict_from_file

# ----------------------------------------------------
# PAGE CONFIGURATION
# ----------------------------------------------------
st.set_page_config(
    page_title="Smart Home Ear",
    page_icon="🔊",
    layout="centered"
)

# ----------------------------------------------------
# HEADER
# ----------------------------------------------------
st.title("🔊 Smart Home Ear")
st.subheader("AI-Based Real-Time Danger Sound Detection")
st.write("Detecting Glass Break, Baby Cry, and Silence.")

# ----------------------------------------------------
# LIVE DETECTION SECTION
# ----------------------------------------------------
st.header("🎧 Live Detection")

status_box = st.empty()
confidence_chart = st.empty()
log_box = st.empty()

prediction_log = []

start_detection = st.checkbox("Start Live Detection")

if start_detection:
    st.success("Model Started... Listening 🔊")

    # Run detection once per rerun (not infinite loop)
    label, probs = predict_sound_live()

    # Status Box
    if label.lower() == "glassbreak":
        status_box.error("🔴 **Glass Break Detected!**")
    elif label.lower() == "babycry":
        status_box.warning("🟡 **Baby Cry Detected**")
    else:
        status_box.success("🟢 **Safe**")

    # Confidence Chart
    confidence_chart.bar_chart({
        "Silence": [probs[0]],
        "Baby Cry": [probs[1]],
        "Glass Break": [probs[2]]
    })

    # Logs
    prediction_log.append(label)
    log_box.write(prediction_log[-5:])

# ----------------------------------------------------
# FILE UPLOAD DETECTION
# ----------------------------------------------------
st.markdown("---")
st.header("📁 Upload Audio File for Detection")

audio_file = st.file_uploader("Upload audio (.wav/.mp3)", type=["wav", "mp3"])

if audio_file:
    with open("uploaded_audio.wav", "wb") as f:
        f.write(audio_file.getbuffer())

    st.success("File uploaded successfully!")

    if st.button("Predict Uploaded File"):
        label, probs = predict_from_file("uploaded_audio.wav")

        st.subheader("Prediction Result")
        st.write(f"**Detected Sound:** {label.upper()}")

        st.subheader("Confidence Scores")
        st.write(f"Silence: {probs[0]:.4f}")
        st.write(f"Baby Cry: {probs[1]:.4f}")
        st.write(f"Glass Break: {probs[2]:.4f}")
