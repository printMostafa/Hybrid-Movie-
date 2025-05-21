import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split

# Page config
st.set_page_config(page_title="Movie Recommender", page_icon="🎬")

# Add CSS
st.markdown("""
    <style>
    .stApp { max-width: 1200px; margin: 0 auto; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        return ratings, movies
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def create_movie_features(movies_data):
    # Process genres
    movies_data['genres'] = movies_data['genres'].fillna('Unknown')
    movies_data['genres'] = movies_data['genres'].apply(lambda x: x.replace('|', ' '))
    
    # Create TF-IDF matrix
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(movies_data['genres'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def train_model(ratings_data):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(ratings_data[['userId', 'movieId', 'rating']], reader)
    trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
    model = SVD()
    model.fit(trainset)
    return model

def get_recommendations(user_id, movie_title, model, movies_df, similarity_matrix, indices):
    # Get movie index
    idx = indices[movie_title]
    
    # Get similarity scores
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Get top 10
    
    # Get movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Get recommendations
    recommendations = movies_df.iloc[movie_indices][['title', 'genres']]
    recommendations['similarity_score'] = [score for _, score in sim_scores]
    
    # Add predicted ratings
    recommendations['predicted_rating'] = recommendations.index.map(
        lambda x: model.predict(user_id, movies_df.iloc[x]['movieId']).est
    )
    
    # Calculate hybrid score
    recommendations['hybrid_score'] = (
        0.4 * recommendations['similarity_score'] + 
        0.6 * recommendations['predicted_rating']
    )
    
    return recommendations.sort_values('hybrid_score', ascending=False)

def main():
    st.title("🎬 Movie Recommendation System")
    
    # Load data
    with st.spinner("Loading data..."):
        ratings, movies = load_data()
        
        if ratings is None or movies is None:
            st.error("Failed to load data files")
            st.stop()
    
    # Process movies
    movies_cleaned = movies.copy()
    similarity_matrix = create_movie_features(movies_cleaned)
    indices = pd.Series(movies_cleaned.index, index=movies_cleaned['title'])
    
    # Train model
    model = train_model(ratings)
    
    # User interface
    col1, col2 = st.columns(2)
    
    with col1:
        user_id = st.number_input(
            "User ID",
            min_value=1,
            max_value=int(ratings['userId'].max()),
            value=1
        )
    
    with col2:
        movie_title = st.selectbox(
            "Select a movie",
            options=movies_cleaned['title'].sort_values()
        )
    
    if st.button("Get Recommendations", type="primary"):
        with st.spinner("Finding movies for you..."):
            recommendations = get_recommendations(
                user_id,
                movie_title,
                model,
                movies_cleaned,
                similarity_matrix,
                indices
            )
            
            st.success("Here are your recommendations:")
            st.dataframe(
                recommendations[['title', 'genres', 'hybrid_score']].round(3),
                hide_index=True
            )

if __name__ == "__main__":
    main()
