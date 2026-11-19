
import streamlit as st
from audio_recorder_streamlit import audio_recorder
import tempfile
import librosa
import numpy as np
import tensorflow as tf


sr= 16000
duration= 5
hop= 160
n_fft= 400
n_mels= 80
n_mfcc = 80
max_len= int((sr*duration) /hop)


audio_bytes= audio_recorder(
    text="Rekam suara",
    recording_color="#0C0C0C",
    neutral_color="#fdfefe"
)

if audio_bytes is None:
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
    f.write(audio_bytes)
    audio_path = f.name

def prepocess_audio(path):
    y, _= librosa.load(path, sr=sr)
    y = librosa.util.fix_length(y, size=sr * duration)

    mel= librosa.feature.melspectrogram(
        y=y,
        sr= sr,
        n_mels= n_mels,
        hop_length=hop,
        n_fft= n_fft,

    )

    mel= librosa.power_to_db(mel)

    mfcc= librosa.feature.mfcc(
        y=y,
        sr= sr,
        n_mfcc= n_mfcc,
        hop_length=hop,
        n_fft= n_fft
    )

    f0, _, _ = librosa.pyin(
        y,
        fmin= librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        hop_length=hop
    )

    f0 = np.nan_to_num(f0)
    f0 = f0[np.newaxis, :]

    mel= mel[:, :max_len]
    mfcc= mfcc[:, :max_len]
    f0= f0[:, :max_len]

    feat = np.vstack([mel, mfcc, f0])     # (161, T)
    feat = feat.T[..., np.newaxis]        # (T, 161, 1)
    feat = feat[np.newaxis, ...]           # (1, T, 161, 1)

    return feat

def load_model():
    return tf.keras.models.load_model("model_crnn.keras")

model= load_model()
x= prepocess_audio(audio_path)
pred= model.predict(x)

label_idx= np.argmax(pred, axis=1)[0]
labels= ["Real", "Synthesis", "Record", "Conversion"]

st.success(f"Hasil deteksi: {labels[label_idx]}")
