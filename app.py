import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# First, check if we can import required libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
except ImportError as e:
    st.error("❌ Error loading required libraries")
    st.error(f"Details: {str(e)}")
    st.info("Please check if all required packages are installed correctly")
    st.stop()

# Add custom CSS
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    try:
        ratings = pd.read_csv('ratings.csv')
        movies = pd.read_csv('movies.csv')
        return ratings, movies
    except Exception as e:
        st.error(f"❌ Error loading data files: {str(e)}")
        return None, None

def process_data(ratings, movies):
    try:
        ratings_with_movies = ratings.merge(movies, on='movieId', how='left')
        ratings_with_movies['genres'] = ratings_with_movies['genres'].fillna('Unknown')
        ratings_with_movies['genres'] = ratings_with_movies['genres'].apply(lambda x: x.replace('|', ' '))
        return ratings_with_movies.drop_duplicates(subset='movieId')[['movieId', 'title', 'genres']]
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

def create_recommendation_model(movies_cleaned):
    try:
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_cleaned['genres'])
        return cosine_similarity(tfidf_matrix, tfidf_matrix)
    except Exception as e:
        st.error(f"❌ Error creating recommendation model: {str(e)}")
        return None

def train_svd_model(ratings):
    try:
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
        trainset, _ = train_test_split(data, test_size=0.2, random_state=42)
        model = SVD()
        model.fit(trainset)
        return model
    except Exception as e:
        st.error(f"❌ Error training SVD model: {str(e)}")
        return None

def hybrid_recommendation(user_id, movie_title, model, movies_cleaned, cosine_sim, indices, weight_cb=0.4, weight_cf=0.6, top_n=10):
    try:
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
        return cb_movies.sort_values('hybrid_score', ascending=False)[['title', 'cb_score', 'cf_score', 'hybrid_score']].head(top_n)
    
    except Exception as e:
        st.error(f"❌ Error generating recommendations: {str(e)}")
        return pd.DataFrame()

# Main app
def main():
    st.title("🎬 Hybrid Movie Recommendation System")
    
    # Load data
    with st.spinner("Loading data..."):
        ratings, movies = load_data()
        if ratings is None or movies is None:
            st.error("Failed to load necessary data files")
            st.stop()
    
    # Process data
    with st.spinner("Processing data..."):
        movies_cleaned = process_data(ratings, movies)
        if movies_cleaned is None:
            st.error("Failed to process data")
            st.stop()
    
    # Create models
    with st.spinner("Creating recommendation models..."):
        cosine_sim = create_recommendation_model(movies_cleaned)
        if cosine_sim is None:
            st.error("Failed to create recommendation model")
            st.stop()
            
        model = train_svd_model(ratings)
        if model is None:
            st.error("Failed to train SVD model")
            st.stop()
    
    indices = pd.Series(movies_cleaned.index, index=movies_cleaned['title']).drop_duplicates()
    
    # User interface
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
        with col2:
            movie_title = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))
    
    if st.button("Show Recommendations", type="primary"):
        with st.spinner("Generating recommendations..."):
            recommendations = hybrid_recommendation(
                user_id, movie_title, model, movies_cleaned, cosine_sim, indices
            )
            
            if not recommendations.empty:
                st.success("📽️ Here are your personalized movie recommendations:")
                
                # Format the scores
                recommendations['cb_score'] = recommendations['cb_score'].round(3)
                recommendations['cf_score'] = recommendations['cf_score'].round(3)
                recommendations['hybrid_score'] = recommendations['hybrid_score'].round(3)
                
                st.dataframe(
                    recommendations,
                    column_config={
                        "title": "Movie Title",
                        "cb_score": "Content Score",
                        "cf_score": "Collaborative Score",
                        "hybrid_score": "Hybrid Score"
                    },
                    hide_index=True
                )

if __name__ == "__main__":
    main()
