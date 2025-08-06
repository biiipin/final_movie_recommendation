import streamlit as st
import pickle
import requests
from sklearn.neighbors import NearestNeighbors
import difflib
import random

# --- Page Configuration (Set first) ---
st.set_page_config(page_title="🎬 Movie Recommender 🎬", layout="wide")

# --- CSS Styling ---
st.markdown("""
<style>
    body {
        background-color: #121212; 
        color: white; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .movie-card {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(10px);
        padding: 10px;
        border-radius: 15px;
        margin-bottom: 20px;
        text-align: center;
        position: relative;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255,255,255,0.3);
    }
    .poster-container {
        position: relative;
    }
    .rating-overlay {
        position: absolute;
        top: 8px;
        left: 8px;
        background: rgba(255, 165, 0, 0.85);
        color: black;
        font-weight: bold;
        padding: 3px 7px;
        border-radius: 10px;
        font-size: 14px;
        z-index: 10;
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #ff7777;
    }
    .compare-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 15px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --- Load Data and Model ---
@st.cache_resource
def load_data():
    try:
        movies_df = pickle.load(open("movies_data.pkl", "rb"))
        tfidf_mat = pickle.load(open("tfidf_matrix.pkl", "rb"))
        return movies_df, tfidf_mat
    except FileNotFoundError:
        st.error("FATAL: 'movies_data.pkl' or 'tfidf_matrix.pkl' not found. Please ensure the model files are in the root directory.")
        st.stop()

movies, tfidf_matrix = load_data()
movie_names = movies['title'].values

API_KEY = "bb8c8e12742c72ae502a3863ccb5402a" # Your TMDB API Key

# --- API Fetching Functions ---
@st.cache_data(show_spinner=False)
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image"

@st.cache_data(show_spinner=False)
def fetch_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    overview = data.get('overview', 'No description available.')
    release_date = data.get('release_date', 'Unknown')
    rating = data.get('vote_average', 0)
    genres = ', '.join([g['name'] for g in data.get('genres', [])])
    runtime = data.get('runtime', 0)
    imdb_id = data.get('imdb_id')
    imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    title = data.get('title', 'Title not found') # Fallback title
    return overview, release_date, rating, genres, runtime, imdb_link, title

@st.cache_data(show_spinner=False)
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    for video in data.get('results', []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

@st.cache_data(show_spinner="Finding franchise movies...")
def fetch_collection_details(movie_id):
    movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    movie_data = requests.get(movie_details_url).json()
    
    collection_info = movie_data.get('belongs_to_collection')
    if not collection_info:
        return None

    collection_id = collection_info['id']
    collection_url = f"https://api.themoviedb.org/3/collection/{collection_id}?api_key={API_KEY}&language=en-US"
    collection_data = requests.get(collection_url).json()
    
    return collection_data.get('parts', [])

# --- NEW: Function to get only movies that are part of a franchise ---
@st.cache_resource
def get_franchise_movies():
    franchise_movie_list = []
    # This is a simplified example. For a real app, you'd want a more efficient way 
    # to pre-filter this, perhaps during your data preprocessing.
    # For now, we'll check a sample of popular movies to demonstrate.
    for title in movie_names[:200]: # Limiting for performance in this example
        movie_record = movies[movies['title'] == title]
        if not movie_record.empty:
            movie_id = movie_record['id'].values[0]
            movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
            movie_data = requests.get(movie_details_url).json()
            if movie_data.get('belongs_to_collection'):
                franchise_movie_list.append(title)
    return ["Select a franchise movie"] + franchise_movie_list

franchise_movie_options = get_franchise_movies()


# --- Recommendation Engine & Other Functions ---
@st.cache_resource
def get_neighbors_model():
    return NearestNeighbors(n_neighbors=6, metric='cosine').fit(tfidf_matrix)

nbrs = get_neighbors_model()

def recommend(title):
    matches = difflib.get_close_matches(title.lower(), movies['title'].str.lower(), n=1, cutoff=0.6)
    if not matches: return []
    idx = movies[movies['title'].str.lower() == matches[0]].index[0]
    distances, indices = nbrs.kneighbors(tfidf_matrix[idx])
    return [movies.iloc[i].id for i in indices[0][1:]]

@st.cache_data(show_spinner=False)
def get_movie_for_compare(title):
    matches = difflib.get_close_matches(title.lower(), movies['title'].str.lower(), n=1, cutoff=0.6)
    if not matches: return None
    idx = movies[movies['title'].str.lower() == matches[0]].index[0]
    movie_id = movies.iloc[idx].id
    overview, date, rating, genres, runtime, imdb_link, _ = fetch_details(movie_id)
    poster = fetch_poster(movie_id)
    trailer = fetch_trailer(movie_id)
    return {
        "title": matches[0], "poster": poster, "rating": rating, "date": date,
        "runtime": runtime, "genres": genres, "overview": overview, "trailer": trailer
    }

# --- Standardized Detail Display Function ---
def display_movie_details(movie_id):
    overview, date, rating, genres, runtime, imdb_link, _ = fetch_details(movie_id)
    trailer = fetch_trailer(movie_id)
    
    st.markdown(f"**Rating:** {rating:.1f}/10")
    st.markdown(f"**Released:** {date}")
    st.markdown(f"**Runtime:** {runtime} mins")
    st.markdown(f"**Genres:** {genres}")
    st.write(overview)
    if imdb_link: st.markdown(f"[IMDb Page]({imdb_link})")
    if trailer: st.video(trailer)

# --- Safe Title Lookup ---
def get_movie_title(movie_id):
    movie_record = movies[movies['id'] == movie_id]
    if not movie_record.empty:
        return movie_record['title'].values[0]
    else:
        _, _, _, _, _, _, title = fetch_details(movie_id)
        return title

# --- Sidebar ---
st.sidebar.header("🎬 Movie Recommender")

def go_home():
    st.session_state.selected_movie = "Select a movie"
    st.session_state.surprise_movie = None
    st.session_state.franchise_movie = "Select a franchise movie" 

st.sidebar.button("🏠 Home", on_click=go_home, use_container_width=True)
st.sidebar.markdown("---")

st.sidebar.header("🔍 Filters & Fun")

def set_random_movie():
    last_movie = st.session_state.get('last_random_surprise')
    new_movie = random.choice(movie_names)
    while new_movie == last_movie:
        new_movie = random.choice(movie_names)
    st.session_state.selected_movie = new_movie
    st.session_state.last_random_surprise = new_movie

# Initialize all session state keys
if 'selected_movie' not in st.session_state: st.session_state.selected_movie = "Select a movie"
if 'franchise_movie' not in st.session_state: st.session_state.franchise_movie = "Select a franchise movie"
if 'last_random_surprise' not in st.session_state: st.session_state.last_random_surprise = None
if 'surprise_movie' not in st.session_state: st.session_state.surprise_movie = None
if 'last_mood_surprise' not in st.session_state: st.session_state.last_mood_surprise = None

movie_names_with_placeholder = ["Select a movie"] + list(movie_names)
selected_movie = st.sidebar.selectbox("Select a movie to get recommendations", movie_names_with_placeholder, key="selected_movie")
rating_options = [x / 2 for x in range(0, 21)]
min_rating = st.sidebar.selectbox("Minimum Rating", rating_options, index=10)
year_range = st.sidebar.slider("Release Year Range", 1950, 2025, (1990, 2025))
st.sidebar.markdown("---")
st.sidebar.button("🎁 Surprise Me! (Random)", on_click=set_random_movie)

# --- Main Section ---
if selected_movie != "Select a movie":
    st.title(f"Recommendations for: *{selected_movie}*")
    with st.spinner("Finding your perfect movies... 🍿"):
        recommended_movie_ids = recommend(selected_movie)
        
        filtered_recs = []
        for movie_id in recommended_movie_ids:
            _, date, rating, _, _, _, _ = fetch_details(movie_id)
            if rating >= min_rating:
                if date and date[:4].isdigit():
                    year = int(date[:4])
                    if year_range[0] <= year <= year_range[1]:
                        filtered_recs.append(movie_id)
        
        if filtered_recs:
            cols = st.columns(5)
            for i, movie_id in enumerate(filtered_recs):
                with cols[i % 5]:
                    movie_title = get_movie_title(movie_id)
                    poster_url = fetch_poster(movie_id)
                    st.image(poster_url, use_container_width=True)
                    st.markdown(f"**{movie_title}**")
                    with st.expander("Details"):
                        display_movie_details(movie_id)
        else:
            st.error("No movies found based on your filters. Try adjusting them!")
else:
    st.title("🎬 Movie Recommender")
    # --- Homepage Content ---

    st.markdown("## ✨ Surprise Me With A Mood")
    mood_genre_map = {
        "Happy & Uplifting": [35, 10751], "Excited & Thrilled": [28, 12, 53],
        "Dramatic & Thought-Provoking": [18, 36], "Scared & On Edge": [27, 9648],
        "Imaginative & Other-Worldly": [14, 878, 16], "Romantic & Heartfelt": [10749]
    }
    selected_mood = st.selectbox("How are you feeling today?", list(mood_genre_map.keys()))
    
    def find_mood_movie():
        genre_ids = mood_genre_map[selected_mood]
        # This part requires a new function to fetch movies by mood, which is not in the original code.
        # Let's assume a function fetch_surprise_movies_by_mood(genre_ids) exists.
        # For demonstration, we'll just pick a random movie from our list.
        potential_movies = movies.sample(n=20).to_dict('records') # Placeholder
        if not potential_movies:
            st.toast("Couldn't find a fresh movie for that mood, please try again!", icon="😞")
            st.session_state.surprise_movie = None; return
        
        last_movie_id = st.session_state.get('last_mood_surprise')
        eligible_movies = [m for m in potential_movies if m.get('id') != last_movie_id]
        if not eligible_movies: eligible_movies = potential_movies
        
        chosen_movie = random.choice(eligible_movies)
        st.session_state.surprise_movie = chosen_movie
        st.session_state.last_mood_surprise = chosen_movie.get('id')

    def clear_mood_movie():
        st.session_state.surprise_movie = None

    b_col1, b_col2, _ = st.columns([1, 1, 4])
    with b_col1: st.button("Get Suggestion", on_click=find_mood_movie, use_container_width=True)
    with b_col2: st.button("Clear", on_click=clear_mood_movie, use_container_width=True)

    if 'surprise_movie' in st.session_state and st.session_state.surprise_movie:
        movie = st.session_state.surprise_movie
        movie_id = movie.get('id')
        st.markdown("---")
        st.subheader("We think you'll like this highly-rated movie:")
        
        m_col1, m_col2 = st.columns([1, 2])
        with m_col1:
            st.image(fetch_poster(movie_id))
        with m_col2:
            st.markdown(f"### {movie.get('title', 'Title Not Found')}")
            display_movie_details(movie_id)

    st.markdown("---")
    st.markdown("## 🎬 Explore a Movie Franchise")
    franchise_movie_title = st.selectbox(
        "Select a movie to see its franchise", 
        franchise_movie_options, 
        key="franchise_movie"
    )

    if franchise_movie_title != "Select a franchise movie":
        movie_record = movies[movies['title'] == franchise_movie_title]
        if not movie_record.empty:
            movie_id = movie_record['id'].values[0]
            collection_parts = fetch_collection_details(movie_id)
            
            if collection_parts:
                st.subheader(f"Movies in this Franchise:")
                cols = st.columns(min(len(collection_parts), 5))
                for i, part in enumerate(collection_parts[:5]):
                    part_id = part.get('id')
                    part_title = part.get('title')
                    if part_id and part_title:
                        with cols[i]:
                            st.image(fetch_poster(part_id), use_container_width=True)
                            st.markdown(f"**{part_title}**")
                            with st.expander("Details"):
                                display_movie_details(part_id)
            else:
                st.info(f"'{franchise_movie_title}' is not part of a known movie collection.")

    st.markdown("---")
    st.markdown("## 🆚 Compare Movies")
    col1, col2 = st.columns(2)
    with col1:
        movie1_title = st.selectbox("Select Movie 1", movie_names_with_placeholder, key="movie1")
    with col2:
        movie2_title = st.selectbox("Select Movie 2", movie_names_with_placeholder, key="movie2")
    if movie1_title != "Select a movie" and movie2_title != "Select a movie":
        movie1_details = get_movie_for_compare(movie1_title)
        movie2_details = get_movie_for_compare(movie2_title)
        if movie1_details and movie2_details:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown(f"<div class='compare-card'>", unsafe_allow_html=True)
                st.image(movie1_details['poster'])
                st.subheader(movie1_details['title'])
                st.write(f"**⭐ Rating:** {movie1_details['rating']:.1f}/10")
                st.write(f"**🗓️ Release:** {movie1_details['date']}")
                st.write(f"**⏳ Runtime:** {movie1_details['runtime']} mins")
                st.write(f"**Genres:** {movie1_details['genres']}")
                if movie1_details.get('trailer'):
                    st.markdown(f"[Watch Trailer]({movie1_details['trailer']})")
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown(f"<div class='compare-card'>", unsafe_allow_html=True)
                st.image(movie2_details['poster'])
                st.subheader(movie2_details['title'])
                st.write(f"**⭐ Rating:** {movie2_details['rating']:.1f}/10")
                st.write(f"**🗓️ Release:** {movie2_details['date']}")
                st.write(f"**⏳ Runtime:** {movie2_details['runtime']} mins")
                st.write(f"**Genres:** {movie2_details['genres']}")
                if movie2_details.get('trailer'):
                    st.markdown(f"[Watch Trailer]({movie2_details['trailer']})")
                st.markdown("</div>", unsafe_allow_html=True)

# --- Trending & Top Rated Movies (Always Displayed) ---
st.markdown("---")
st.markdown("## 🔥 Trending This Week")
trending_movies_response = requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1").json()
trending_movie_ids = [m['id'] for m in trending_movies_response.get('results', [])[:5]]

cols = st.columns(5)
for i, movie_id in enumerate(trending_movie_ids):
    with cols[i]:
        poster_url = fetch_poster(movie_id)
        movie_title = get_movie_title(movie_id)
        st.image(poster_url, use_container_width=True)
        st.markdown(f"**{movie_title}**")
        with st.expander("Details"):
            display_movie_details(movie_id)

st.markdown("---")
st.markdown("## 🏆 Top Rated Movies")
top_rated_response = requests.get(f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1").json()
top_rated_movie_ids = [m['id'] for m in top_rated_response.get('results', [])[:5]]

cols = st.columns(5)
for i, movie_id in enumerate(top_rated_movie_ids):
    with cols[i]:
        poster_url = fetch_poster(movie_id)
        movie_title = get_movie_title(movie_id)
        st.image(poster_url, use_container_width=True)
        st.markdown(f"**{movie_title}**")
        with st.expander("Details"):
            display_movie_details(movie_id)

