import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Load data
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')
ratings_with_movies = ratings.merge(movies, on='movieId', how='left')
ratings_with_movies['genres'] = ratings_with_movies['genres'].fillna('Unknown')
ratings_with_movies['genres'] = ratings_with_movies['genres'].apply(lambda x: x.replace('|', ' '))
movies_cleaned = ratings_with_movies.drop_duplicates(subset='movieId')[['movieId', 'title', 'genres']]

# TF-IDF Processing
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_cleaned['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['title']).drop_duplicates()

# Train SVD model
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)
model = SVD()
model.fit(trainset)

# Hybrid recommendation function
def hybrid_recommendation(user_id, movie_title, model, ratings_df, movies_df, cosine_sim, weight_cb=0.4, weight_cf=0.6, top_n=10):
    if movie_title not in indices:
        return pd.DataFrame([{"title": "Movie not found"}])
    
    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    movie_indices = [i[0] for i in sim_scores[1:top_n+30]]
    cb_movies = movies_cleaned.iloc[movie_indices][['movieId', 'title']].copy()
    cb_movies['cb_score'] = [score for _, score in sim_scores[1:top_n+30]]
    
    cf_scores = []
    for _, row in cb_movies.iterrows():
        pred = model.predict(user_id, row['movieId'])
        cf_scores.append(pred.est)
    cb_movies['cf_score'] = cf_scores
    
    cb_movies['hybrid_score'] = (cb_movies['cb_score'] * weight_cb) + (cb_movies['cf_score'] * weight_cf)
    cb_movies = cb_movies.sort_values('hybrid_score', ascending=False)
    return cb_movies[['title', 'cb_score', 'cf_score', 'hybrid_score']].head(top_n).reset_index(drop=True)

# Streamlit UI
st.title("🎬 Hybrid Movie Recommendation System")

user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
movie_title = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))

if st.button("Show Recommendations"):
    st.write("📽️ Hybrid Recommendations:")
    recommendations = hybrid_recommendation(user_id, movie_title, model, ratings, movies, cosine_sim)
    st.dataframe(recommendations)
