import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

# 1. Load Data
ratings = pd.read_csv('ratings.csv')  
movies = pd.read_csv('movies.csv')    

# 2. Quick Data Overview
print("Ratings shape:", ratings.shape)
print("Movies shape:", movies.shape,"\n")

# 3. Check Missing Values
print(ratings.isnull().sum())
print(movies.isnull().sum(),"\n")

# 4. Merge Data as Needed
ratings_with_movies = ratings.merge(movies, on='movieId', how='left')

# 5. Process Columns (e.g., genres separated by "|")
ratings_with_movies['genres'] = ratings_with_movies['genres'].fillna('Unknown')
ratings_with_movies['genres'] = ratings_with_movies['genres'].apply(lambda x: x.replace('|', ' '))

# Final Preview
print(ratings_with_movies.head(),"\n")
movies_cleaned = ratings_with_movies.drop_duplicates(subset='movieId')[['movieId', 'title', 'genres']]

# 1. TF-IDF Vectorization on genres
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_cleaned['genres'])

print("TF-IDF Matrix shape:", tfidf_matrix.shape,"\n")  # Number of movies × Number of unique words in genres

# 2. Calculate Cosine Similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# 3. Build mapping from movie title to DataFrame index
indices = pd.Series(movies_cleaned.index, index=movies_cleaned['title']).drop_duplicates()

# 4. Recommendation Function
def recommend_movies(title, cosine_sim=cosine_sim):
    if title not in indices:
        return "Movie not found in the database."
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Top 10 similar movies

    movie_indices = [i[0] for i in sim_scores]
    return movies_cleaned['title'].iloc[movie_indices]

# ✅ Test Recommendation
print(recommend_movies('Toy Story (1995)'),"\n")

# 1. Prepare Data from Original Ratings File
reader = Reader(rating_scale=(0.5, 5.0))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

# 2. Split Data into Train/Test
trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

# 3. Build and Train Model
model = SVD()
model.fit(trainset)

# 4. Evaluate Model
predictions = model.test(testset)
rmse = accuracy.rmse(predictions)
mae = accuracy.mae(predictions)

print(f"✅ RMSE: {rmse:.4f}, MAE: {mae:.4f}","\n")

def recommend_for_user(user_id, movies_df, model, ratings_df, top_n=10):
    # Movies the user has watched
    watched_movies = ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist()
    all_movies = movies_df['movieId'].tolist()
    unseen_movies = [m for m in all_movies if m not in watched_movies]

    # Predict ratings
    predictions = [model.predict(user_id, movie_id) for movie_id in unseen_movies]
    predictions.sort(key=lambda x: x.est, reverse=True)

    # Extract top n movies
    top_movies_ids = [pred.iid for pred in predictions[:top_n]]
    recommended_titles = movies_df[movies_df['movieId'].isin(top_movies_ids)]['title']

    return recommended_titles.reset_index(drop=True)

# ✅ Test User Recommendation for User 1
print(recommend_for_user(1, movies, model, ratings),"\n")

def hybrid_recommendation(user_id, movie_title, model, ratings_df, movies_df, cosine_sim, weight_cb=0.4, weight_cf=0.6, top_n=10):
    # Step 1: Content-Based predictions
    if movie_title not in indices:
        return "Movie not found in the database."

    idx = indices[movie_title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    movie_indices = [i[0] for i in sim_scores[1:top_n+30]]  # Get more than top_n to filter later
    cb_movies = movies_cleaned.iloc[movie_indices][['movieId', 'title']].copy()
    cb_movies['cb_score'] = [score for _, score in sim_scores[1:top_n+30]]

    # Step 2: Collaborative Filtering prediction
    cf_scores = []
    for _, row in cb_movies.iterrows():
        pred = model.predict(user_id, row['movieId'])
        cf_scores.append(pred.est)

    cb_movies['cf_score'] = cf_scores

    # Step 3: Weighted Hybrid Score
    cb_movies['hybrid_score'] = (cb_movies['cb_score'] * weight_cb) + (cb_movies['cf_score'] * weight_cf)

    # Step 4: Return top N
    cb_movies = cb_movies.sort_values('hybrid_score', ascending=False)
    return cb_movies[['title', 'cb_score', 'cf_score', 'hybrid_score']].head(top_n).reset_index(drop=True)

# ✅ Test Hybrid Recommendation
print("🎬 Hybrid recommendations for user 1 based on 'Toy Story (1995)':\n")
print(hybrid_recommendation(1, 'Toy Story (1995)', model, ratings, movies, cosine_sim))
