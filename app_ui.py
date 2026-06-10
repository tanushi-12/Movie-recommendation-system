import streamlit as st
import pickle
import requests
import torch
import clip
from PIL import Image
import numpy as np

OMDB_API_KEY = "d521b86c"

st.set_page_config(page_title="Cinematic AI Engine", layout="wide")
st.title("🎬 Cinematic AI Recommendation Engine")


@st.cache_resource
def load_data():
    movies = pickle.load(open("movies.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    clip_embeddings = pickle.load(open("clip_text_embeddings.pkl", "rb"))
    return movies, similarity, clip_embeddings

movies, similarity, clip_text_embeddings = load_data()
 

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_text_embeddings = clip_text_embeddings.to(device)


if "watchlist" not in st.session_state:
    st.session_state.watchlist = []


@st.cache_data
def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}"
    return requests.get(url).json()


def recommend(movie_name):
    movies['title_lower'] = movies['title'].str.lower()
    index = movies[movies['title_lower'] == movie_name.lower()].index[0]
    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in sorted_movies]


selected_movie = st.selectbox("Choose a movie", movies['title'].values)

if st.button("Recommend"):
    st.session_state.recommended = recommend(selected_movie)

if "recommended" in st.session_state:
    cols = st.columns(5)

    for idx, name in enumerate(st.session_state.recommended):
        data = fetch_movie_details(name)

        with cols[idx]:
            if data.get("Poster") and data["Poster"] != "N/A":
                st.image(data["Poster"], use_container_width=True)

            st.markdown(f"**{name}**")

            if st.button("❤️ Add", key=f"watch_{idx}"):
                if name not in st.session_state.watchlist:
                    st.session_state.watchlist.append(name)

            if st.button("View Details", key=f"details_{idx}"):
                st.session_state.popup = data


if "popup" in st.session_state:

    data = st.session_state.popup

    @st.dialog(data.get("Title"))
    def show_popup():
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(data.get("Poster"), use_container_width=True)

        with col2:
            st.markdown(f"## {data.get('Title')}")
            st.write(f"⭐ IMDb: {data.get('imdbRating')}")
            st.write(f"📅 Year: {data.get('Year')}")
            st.write("---")
            st.write(data.get("Plot"))

    show_popup()


st.markdown("---")
st.subheader("🎥 Scene Recognition AI")

uploaded_file = st.file_uploader("Upload a movie scene image", type=["jpg", "png"])

if uploaded_file:

    col1, col2 = st.columns([1, 2])

    with col1:
        st.image(uploaded_file, caption="Uploaded Image", width=250)

    with col2:
        if st.button("🔍 Analyze Scene"):

            with st.spinner("Analyzing scene..."):
                image = preprocess(Image.open(uploaded_file)).unsqueeze(0).to(device)

                with torch.no_grad():
                    image_features = clip_model.encode_image(image)
                    image_features /= image_features.norm(dim=-1, keepdim=True)

                    similarities = (image_features @ clip_text_embeddings.T).cpu().numpy()

                best_match_index = np.argmax(similarities)
                predicted_movie = movies.iloc[best_match_index].title
                confidence = float(np.max(similarities) * 100)

            st.success(f"🎬 Predicted Movie: {predicted_movie}")
            st.info(f"Confidence Score: {confidence:.2f}%")


st.markdown("---")
st.subheader("📌 Your Watchlist")

if st.session_state.watchlist:
    for movie in st.session_state.watchlist:
        st.write("•", movie)
else:
    st.write("No movies added yet.")
