import streamlit as st

# Add error handling for imports
try:
    import pandas as pd
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from surprise import Dataset, Reader, SVD
    from surprise.model_selection import train_test_split
except ImportError as e:
    st.error(f"Failed to import required libraries. Error: {str(e)}")
    st.info("Please make sure all required libraries are installed. Check requirements.txt")
    st.stop()

# Page config
st.set_page_config(
    page_title="Hybrid Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

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
        st.error(f"Error loading data: {str(e)}")
        return None, None

# Load data
try:
    ratings, movies = load_data()

    if ratings is not None and movies is not None:
        # Process data
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

        def hybrid_recommendation(user_id, movie_title, model, ratings_df, movies_df, cosine_sim, weight_cb=0.4, weight_cf=0.6, top_n=10):
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
                cb_movies = cb_movies.sort_values('hybrid_score', ascending=False)
                return cb_movies[['title', 'cb_score', 'cf_score', 'hybrid_score']].head(top_n).reset_index(drop=True)
            except Exception as e:
                st.error(f"Error in recommendation: {str(e)}")
                return pd.DataFrame()

        # Streamlit UI
        st.title("🎬 Hybrid Movie Recommendation System")
        
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                user_id = st.number_input("Enter User ID:", min_value=1, max_value=int(ratings['userId'].max()), value=1)
            
            with col2:
                movie_title = st.selectbox("Select a movie you like:", sorted(movies['title'].unique()))

        if st.button("Show Recommendations", type="primary"):
            with st.spinner("Generating recommendations..."):
                recommendations = hybrid_recommendation(user_id, movie_title, model, ratings, movies, cosine_sim)
                
                if not recommendations.empty:
                    st.success("📽️ Here are your personalized movie recommendations:")
                    
                    # Format the scores to be more readable
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
    else:
        st.error("Failed to load the required data files. Please make sure 'ratings.csv' and 'movies.csv' are present in the directory.")
except Exception as e:
    st.error(f"An error occurred while running the application: {str(e)}")
    st.info("Please check the logs for more details and make sure all requirements are installed correctly.")
