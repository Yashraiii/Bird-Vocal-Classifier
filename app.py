import streamlit as st
from backend.serving import serving
import numpy as np
import tensorflow as tf
import tensorflow_io as tfio


# Heading of the app
st.title("Bird Vocal Classifier")

def frame_audio(
      audio_array: np.ndarray,
      window_size_s: float = 5.0,
      hop_size_s: float = 5.0,
      sample_rate = 32000,
  ) -> np.ndarray:
    """Helper function for framing audio for inference."""
    if window_size_s is None or window_size_s < 0:
      return audio_array[np.newaxis, :]
    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)
    return framed_audio

def ensure_sample_rate(waveform, original_sample_rate,
                       desired_sample_rate=32000):
  """Resample waveform if required."""
  if original_sample_rate != desired_sample_rate:
    waveform = tfio.audio.resample(waveform, original_sample_rate, desired_sample_rate)
  return desired_sample_rate, waveform




#Taking Audio Input file from user
uploaded_file = st.file_uploader("Choose an audio file", type="wav")
# convert wav to array
if uploaded_file is not None:
    audio_bytes = uploaded_file.read()
    st.write(type(audio_bytes))
    st.write(len(audio_bytes))
    audio_array = np.frombuffer(audio_bytes, dtype=np.uint8)
    st.write(type(audio_array))
    st.write(len(audio_array))
    st.write(audio_array)
    st.write(audio_array.shape)
    print("following is the shape of audio array")
    fixed_tm = frame_audio(audio_array, 5.0, 5.0,32000)
    fixed_tm.shape













