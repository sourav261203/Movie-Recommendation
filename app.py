import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import pickle

# --- Setup Page ---
st.set_page_config(page_title="Movie Recommender", layout="wide", page_icon="🎬")

# --- Load Data ---
@st.cache_data
def load_data():
    df_popular = pd.DataFrame(pickle.load(open('df_popular.pkl', 'rb')))
    df_recommend = pd.DataFrame(pickle.load(open('df_recommend.pkl', 'rb')))
    similarity = pickle.load(open('similarity.pkl', 'rb'))
    return df_popular, df_recommend, similarity

df_popular, df_recommend, similarity = load_data()

# Helper to fetch poster URL
def fetch_poster(poster_path):
    if pd.isna(poster_path):
        return "https://via.placeholder.com/500x750?text=No+Poster+Available"
    return "https://image.tmdb.org/t/p/w500/" + poster_path

# Helper to format lists for UI display
def format_list(item_list):
    if isinstance(item_list, list):
        return ", ".join(item_list)
    return str(item_list)

# --- Navigation Menu ---
selected = option_menu(
    menu_title=None,
    options=['Recommend Me a Movie', 'Describe a Movie', 'Top Popular Movies'],
    icons=['film', 'info-circle', 'fire'],
    menu_icon='cast',
    default_index=0,
    orientation="horizontal",
)

# ==========================================
# SECTION 1: RECOMMEND ME A MOVIE
# ==========================================
if selected == 'Recommend Me a Movie':
    st.title('🎯 Top 10 Recommended Movies')
    
    selected_movie = st.selectbox('Type or select a movie you like:', df_recommend['title'].values)
    
    if st.button('Recommend'):
        # Find movie index
        movie_index = df_recommend[df_recommend['title'] == selected_movie].index[0]
        distances = similarity[movie_index]
        # Get top 10 movies (skipping the first one which is the movie itself)
        movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:11]
        
        # Display 2 rows of 5 movies
        for row in range(2):
            cols = st.columns(5)
            for col_idx in range(5):
                movie_idx = movies_list[row * 5 + col_idx][0]
                movie_title = df_recommend.iloc[movie_idx].title
                poster = fetch_poster(df_recommend.iloc[movie_idx].poster_path)
                
                with cols[col_idx]:
                    st.image(poster, use_container_width=True)
                    st.write(f"**{movie_title}**")

# ==========================================
# SECTION 2: DESCRIBE A MOVIE
# ==========================================
elif selected == 'Describe a Movie':
    st.title('🎞️ Movie Description')
    
    selected_movie = st.selectbox('Select a movie to see details:', df_recommend['title'].values)
    
    if selected_movie:
        movie_data = df_recommend[df_recommend['title'] == selected_movie].iloc[0]
        
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(fetch_poster(movie_data['poster_path']), use_container_width=True)
            
        with col2:
            st.header(movie_data['title'])
            st.write(f"**📅 Release Date:** {pd.to_datetime(movie_data['release_date']).date() if not pd.isna(movie_data['release_date']) else 'N/A'}")
            st.write(f"**⏱️ Runtime:** {movie_data['runtime']} minutes")
            st.write(f"**⭐ Average Rating:** {movie_data['averageRating']}/10")
            
            st.subheader("Overview")
            st.write(movie_data['overview'])
            
            # Using the processed list data
            st.write(f"**🎭 Genres:** {format_list(df_popular[df_popular['title'] == selected_movie].iloc[0]['genres'])}")
            st.write(f"**🎬 Directors:** {format_list(movie_data['directors'])}")
            st.write(f"**👥 Top Cast:** {format_list(movie_data['cast'][:5])}")

# ==========================================
# SECTION 3: TOP POPULAR MOVIES (FILTERS)
# ==========================================
elif selected == 'Top Popular Movies':
    st.title('🔥 Top Popular Movies')
    
    # 1. Sidebar Filters
    st.sidebar.header("Filter Movies")
    
    # Extract unique genres and countries for multiselect
    all_genres = sorted(list(set([g for sublist in df_popular['genres'] for g in sublist if g])))
    all_countries = sorted(list(set([c for sublist in df_popular['production_countries'] for c in sublist if c])))
    
    selected_genres = st.sidebar.multiselect("Select Genres", all_genres)
    selected_countries = st.sidebar.multiselect("Production Countries", all_countries)
    
    # Year Slider
    min_year = int(df_popular['release_date'].dt.year.min())
    max_year = int(df_popular['release_date'].dt.year.max())
    year_range = st.sidebar.slider("Release Year Range", min_year, max_year, (2000, max_year))
    
    # Runtime Slider
    max_runtime = int(df_popular['runtime'].max())
    runtime_range = st.sidebar.slider("Runtime Range (minutes)", 0, max_runtime, (60, 200))
    
    # Top N slider
    top_n = st.sidebar.slider("Number of movies to show", 10, 50, 20)

    # 2. Filtering Logic (Adapted from Notebook)
    def filter_popular_movies(df, start_yr, end_yr, run_low, run_high, genres, countries, n):
        filtered = df.copy()
        
        # Year and Runtime Filter
        filtered = filtered[
            (filtered["release_date"].dt.year >= start_yr) & 
            (filtered["release_date"].dt.year <= end_yr) &
            (filtered["runtime"] >= run_low) & 
            (filtered["runtime"] <= run_high)
        ]
        
        # Genre Filter
        if genres:
            filtered = filtered[filtered["genres"].apply(lambda x: any(g in x for g in genres))]
            
        # Country Filter
        if countries:
            filtered = filtered[filtered["production_countries"].apply(lambda x: any(c in x for c in countries))]
            
        # Minimum constraints to ensure quality popular movies
        filtered = filtered[(filtered["numVotes"] >= 5000) & (filtered["popularity"] >= 10)]
        
        # Sort by Rating and Votes
        filtered = filtered.sort_values(by=["averageRating", "numVotes"], ascending=False)
        return filtered.head(n)

    # Apply filters
    results = filter_popular_movies(
        df_popular, 
        start_yr=year_range[0], end_yr=year_range[1], 
        run_low=runtime_range[0], run_high=runtime_range[1], 
        genres=selected_genres, countries=selected_countries, 
        n=top_n
    )

    # 3. Display Results
    if results.empty:
        st.warning("No movies found matching your criteria. Try adjusting the filters!")
    else:
        st.success(f"Found {len(results)} movies matching your criteria!")
        
        # Display in a dynamic grid
        num_cols = 4
        for i in range(0, len(results), num_cols):
            cols = st.columns(num_cols)
            for j in range(num_cols):
                if i + j < len(results):
                    movie = results.iloc[i + j]
                    with cols[j]:
                        st.image(fetch_poster(movie['poster_path']), use_container_width=True)
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"⭐ {movie['averageRating']} | 🗓️ {movie['release_date'].year}")
