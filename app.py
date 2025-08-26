import streamlit as st
import librosa
import numpy as np
from model import EmotionAnalyzer
import os
from dotenv import load_dotenv
from utils import process_audio, plot_emotions

# Load environment variables
load_dotenv()

def main():
    st.title("Audio Emotion Analyzer")
    st.write("Upload an audio file to analyze the emotions in it!")

    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=['wav', 'mp3'])

    if uploaded_file is not None:
        st.audio(uploaded_file)
        
        # Create a progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Process the audio file
        status_text.text("Processing audio file...")
        features = process_audio(uploaded_file)
        progress_bar.progress(50)
        
        # Load model and predict
        status_text.text("Analyzing emotions...")
        analyzer = EmotionAnalyzer()
        emotions = analyzer.predict(features)
        progress_bar.progress(100)
        
        # Display results
        status_text.text("Analysis complete!")
        
        # Plot emotions
        st.subheader("Emotion Analysis Results")
        fig = plot_emotions(emotions)
        st.pyplot(fig)
        
        # Display emotion percentages
        st.subheader("Emotion Percentages")
        for emotion, score in emotions.items():
            st.write(f"{emotion}: {score:.2f}%")

if __name__ == "__main__":
    main()