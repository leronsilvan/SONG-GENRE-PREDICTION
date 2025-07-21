import librosa
import numpy as np
import pandas as pd

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path, mono=True)

    features = {
        'danceability': np.mean(librosa.onset.onset_strength(y=y, sr=sr)),
        'energy': np.mean(librosa.feature.rms(y=y)),
        'acousticness': np.mean(librosa.feature.spectral_flatness(y=y)),
        'instrumentalness': np.mean(librosa.feature.zero_crossing_rate(y)),
        'liveness': np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)),
        'valence': np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)),
        'tempo': librosa.beat.tempo(y=y, sr=sr)[0],
        'speechiness': np.mean(librosa.feature.mfcc(y=y, sr=sr)[0]),
    }

    return pd.DataFrame([features])