# Movie-recommendation-
# 🎬 Cinematic AI Recommendation Engine

A movie recommendation system built using Python, Streamlit, Machine Learning, and OpenAI CLIP.

## Features

* 🎥 Content-Based Movie Recommendation
* ❤️ Watchlist Management
* ℹ️ Movie Details Popup
* 🖼️ Scene Recognition using CLIP
* ⭐ IMDb Ratings, Genre, Year, and Plot Information
* 🎨 Interactive Streamlit User Interface

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* PyTorch
* OpenAI CLIP
* OMDb API

## Project Structure

movie/
│
├── app.py
├── app_ui.py
├── model_builder.py
├── movies.pkl
├── similarity.pkl
├── clip_text_embeddings.pkl
├── tmdb_5000_movies.csv
├── tmdb_5000_credits.csv
├── requirements.txt
└── README.md

## How It Works

### Recommendation Engine

The system recommends movies using content-based filtering. Similarity scores are computed from movie metadata and used to find movies that are most similar to the selected movie.

### Scene Recognition

Users can upload a movie scene image. OpenAI CLIP extracts image features and compares them with precomputed movie embeddings to predict the most likely movie.

## Installation

```bash
git clone https://github.com/tanushi-12/Movie-recommendation-system.git
cd Movie-recommendation-system
pip install -r requirements.txt
streamlit run app_ui.py
```

## Future Improvements

* Poster-based image embeddings
* Advanced watchlist management
* User authentication
* Hybrid recommendation system
* Deployment on Streamlit Cloud

## Author

Tanushi Taparia
