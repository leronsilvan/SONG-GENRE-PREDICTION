import streamlit as st
import pickle
from extract_features import extract_audio_features

# Load the trained model
model = pickle.load(open("genre_model.pkl", "rb"))

# Genre label mapping
genre_mapping = {
    0: "High-Energy Rock",
    1: "Dance Pop",
    2: "Mainstream Pop",
    3: "Funky Pop",
    4: "Indie Pop",
    5: "Electro Pop",
    6: "Modern Dance",
    7: "Mainstream Rock",
    8: "Low-Energy Rock",
    9: "Alt Rock / Synth Pop",
    10: "Chill Acoustic",
    11: "Instrumental Piano",
    12: "Happy Pop",
    13: "Popular Acoustic",
    14: "Ambient Instrumental",
    15: "Classical / Ambient",
    16: "Top Hits / Dance Hits",
    17: "Hard Rock / Metal"
}

st.title("🎵 Audio Genre Classifier")

uploaded_file = st.file_uploader("Upload an audio file (.mp3 or .wav)", type=['mp3', 'wav'])

if uploaded_file is not None:
    st.audio(uploaded_file)

    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    st.info("Extracting audio features...")
    features = extract_audio_features("temp_audio.wav")

    if features is not None:
        st.write("### Extracted Features", features)

        st.success("Classifying...")
        prediction = model.predict(features)[0]

        # Get genre label
        genre = genre_mapping.get(prediction, "Unknown Genre")
        st.subheader(f"🎧 Predicted Genre: `{genre}`")
    else:
        st.error("❌ Failed to extract features. Please upload a valid audio file.")