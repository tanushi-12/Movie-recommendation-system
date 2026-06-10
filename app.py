import streamlit as st
import pickle
import requests

OMDB_API_KEY = "d521b86c"

st.set_page_config(page_title="Movie Recommender", layout="wide")

st.title("🎬 Advanced Movie Recommendation System")

@st.cache_resource
def load_model():
    movies = pickle.load(open("movies.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity

movies, similarity = load_model()

def fetch_poster(movie_title):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={movie_title}"
    response = requests.get(url)
    data = response.json()

    if data.get("Poster") and data["Poster"] != "N/A":
        return data["Poster"]
    return None

def recommend(movie_name):
    movies['title_lower'] = movies['title'].str.lower()

    index = movies[movies['title_lower'] == movie_name.lower()].index[0]

    distances = list(enumerate(similarity[index]))
    sorted_movies = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]

    recommended_movies = []
    posters = []

    for i in sorted_movies:
        title = movies.iloc[i[0]].title
        recommended_movies.append(title)
        posters.append(fetch_poster(title))

    return recommended_movies, posters

selected_movie = st.selectbox(
    "Select a movie",
    movies['title'].values
)

if st.button("Recommend"):
    names, posters = recommend(selected_movie)

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.markdown(f"**{names[i]}**")
            if posters[i]:
                st.image(posters[i])
            else:
                st.write("Poster not available")

    
