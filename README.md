#  Audio Genre Classifier

This is a Streamlit-based web app that classifies audio files into music genres using audio features like danceability, energy, acousticness, tempo, etc. It uses a trained machine learning model (`RandomForestClassifier`) to predict the genre.

 # Features
 
- Upload `.mp3` or `.wav` files
- Extract audio features using `librosa`
- Predicts genre from 18 categories (e.g., Rock, Pop, Acoustic, etc.)
- User-friendly interface built with Streamlit