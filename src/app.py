import streamlit as st
import pickle
import librosa
import librosa.display
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import os
import feature_extraction
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

genre_info = {
    "blues": "A music genre characterized by its 12-bar blues chord progression.",
    "classical": "A genre rooted in Western art music traditions, known for orchestras and complex structures.",
    "country": "A genre centered around storytelling, often featuring acoustic instruments.",
    "disco": "A dance-oriented genre with steady beats and groovy basslines.",
    "hiphop": "A genre combining rhythmic beats with rapping and lyrical storytelling.",
    "jazz": "A genre known for its improvisation and swing.",
    "metal": "A powerful genre with distorted guitars, intense vocals, and heavy drumming.",
    "pop": "Popular music known for its catchy melodies and wide appeal.",
    "reggae": "A Jamaican music style known for its relaxed rhythms and offbeat accents.",
    "rock": "A high-energy genre featuring electric guitars, strong beats, and bold vocals."
}

src_dir = os.path.dirname(os.path.abspath(__file__))

base_dir = os.path.abspath(os.path.join(src_dir, ".."))

labelencoder_path = os.path.join(base_dir, "data", "processed", "labelencoder.pkl")
standardscaler_path = os.path.join(base_dir, "data", "processed", "standardscaler.pkl")
pca_path = os.path.join(base_dir, "data", "processed", "pca.pkl")
best_svm_path = os.path.join(base_dir, "data", "processed", "best_svm.pkl")
csv_file = os.path.join(base_dir, "data", "processed", "musicgenre.csv")
genres_original_path = os.path.join(base_dir, "data", "raw", "genres_original")

# Load models
with open(labelencoder_path, "rb") as file:
    labelencoder = pickle.load(file)
with open(standardscaler_path, "rb") as file:
    scaler = pickle.load(file)
with open(pca_path, "rb") as file:
    pca = pickle.load(file)
with open(best_svm_path, "rb") as file:
    classifier = pickle.load(file)
    
def find_similar_songs(uploaded_file):
    df = pd.read_csv(csv_file)
    
    y, sr = librosa.load(uploaded_file, sr=None)
    uploaded_features = feature_extraction.get_features(y, sr).reshape(1, -1)  

    features = df.iloc[:, 1:-1].values 
    filenames = df['filename']
    

    similarities = cosine_similarity(uploaded_features, features).flatten()

    sorted_indices = np.argsort(similarities)[::-1]

    for idx in sorted_indices:
        if similarities[idx] < 0.99: 
            similar_song = filenames.iloc[idx]
            similarity_score = similarities[idx]
            return similar_song, similarity_score

    return None, None



# Streamlit app
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('https://cdn.pixabay.com/photo/2020/11/02/05/56/music-5705801_1280.jpg');
        background-size: cover;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .stAlert {
        background-color: rgba(0,128,0, 0.9); 
        border-radius: 15px; /* Rounded corners */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .popup {
        position: fixed;
        top: 20%;
        right: 2%;
        background-color: #f0f8ff; /* Light blue */
        border: 1px solid #0078d4; /* Dark blue border */
        border-radius: 8px; /* Slightly smaller radius */
        padding: 10px; /* Reduced padding */
        font-family: 'Arial', sans-serif;
        font-size: 14px; /* Smaller font size */
        color: #004080; /* Dark blue text */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        z-index: 1000;
        animation: fadeIn 0.5s;
        width: 250px;  /* Smaller width */
        height: auto;  /* Let height adjust based on content */
    }

    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateX(50%);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Audio Genre Classification and Recommendation")

st.sidebar.header("Settings")
segment_duration = st.sidebar.slider("Segment Duration (seconds)", min_value=1, max_value=10, value=3)
st.sidebar.write(f"Current segment duration: {segment_duration} seconds")


uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])


if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    st.write("Processing audio...")

    try:
        y, sr = librosa.load(uploaded_file, sr=None)
        features = feature_extraction.get_features(y, sr)
        
        # Predict Genre
        scaled_features = scaler.transform(features.reshape(1, -1))
        reduced_features = pca.transform(scaled_features)
        prediction = classifier.predict(reduced_features)
        predicted_label = labelencoder.inverse_transform([prediction[0]])[0]

        st.success(f"Predicted Genre: {predicted_label.upper()}")
        genre_description = genre_info.get(predicted_label, 'No information available.')
        st.markdown(
            f"""
            <div class="popup">
                <strong>About {predicted_label.upper()}:</strong><br>
                {genre_description}
            </div>
            """,
            unsafe_allow_html=True,
        )

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=np.linspace(0, len(y)/sr, len(y)), y=y, mode='lines', name="Waveform"))
        fig.update_layout(
            title="Waveform of Audio",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
            width=800,  
            height=400,  
        )
        st.plotly_chart(fig)
        
        # Recommend
        if st.button("Recommend"):
            temp_file_path = "temp_uploaded_file.wav"
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            similar_song, similarity = find_similar_songs(temp_file_path)
            
            st.success(f"🎵 Most similar song: **{similar_song}**   Similarity score: **{similarity:.2f}**")
            
            genres_dir = genres_original_path 
            folder_name = similar_song.split('.')[0] 
            song_path = os.path.join(genres_dir, folder_name, similar_song)
            
            st.audio(song_path)
        
    except Exception as e:
        st.error(f"An error occurred: {e}")