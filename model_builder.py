import pandas as pd
import pickle
import torch
import clip
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("Loading datasets...")

movies = pd.read_csv("tmdb_5000_movies.csv")
credits = pd.read_csv("tmdb_5000_credits.csv")

movies = movies.merge(credits, on='title')
movies = movies[['title', 'overview']]
movies.dropna(inplace=True)

print("Creating TF-IDF vectors...")

tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
vectors = tfidf.fit_transform(movies['overview'])
similarity = cosine_similarity(vectors)

pickle.dump(movies, open("movies.pkl", "wb"))
pickle.dump(similarity, open("similarity.pkl", "wb"))

print("TF-IDF similarity saved.")


print("Loading CLIP model...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

texts = movies['overview'].tolist()
text_tokens = clip.tokenize(texts, truncate=True).to(device)

print("Encoding text embeddings...")

with torch.no_grad():
    text_features = model.encode_text(text_tokens)
    text_features /= text_features.norm(dim=-1, keepdim=True)

pickle.dump(text_features.cpu(), open("clip_text_embeddings.pkl", "wb"))

print("CLIP embeddings saved.")
print("Model building complete.")
