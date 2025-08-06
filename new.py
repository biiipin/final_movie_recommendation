import streamlit as st
import pickle
import requests
import difflib
import random
from sklearn.neighbors import NearestNeighbors

st.set_page_config(page_title="🎬 Movie Recommender 🎬", layout="wide")

st.markdown("""
<style>
    body {
        background-color: #121212; 
        color: white; 
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
    }
    .stExpander header {
        font-size: 1.5rem;
        font-weight: bold;
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
        height: 100%;
    }
</style>
""", unsafe_allow_html=True)

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

API_KEY = "bb8c8e12742c72ae502a3863ccb5402a"


@st.cache_data(show_spinner="Fetching available genres...")
def fetch_genres():
    url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={API_KEY}&language=en-US"
    try:
        response = requests.get(url)
        response.raise_for_status()
        genres = response.json().get('genres', [])
        return {genre['name']: genre['id'] for genre in genres}
    except requests.exceptions.RequestException:
        return {}

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
    
    details = {
        'overview': data.get('overview', 'No description available.'),
        'release_date': data.get('release_date', 'Unknown'),
        'rating': data.get('vote_average', 0),
        'genres': ', '.join([g['name'] for g in data.get('genres', [])]),
        'runtime': data.get('runtime', 0),
        'imdb_id': data.get('imdb_id'),
        'imdb_link': f"https://www.imdb.com/title/{data.get('imdb_id')}/" if data.get('imdb_id') else None,
        'title': data.get('title', 'Title not found')
    }
    return details

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
    if not collection_info: return None
    collection_id = collection_info['id']
    collection_url = f"https://api.themoviedb.org/3/collection/{collection_id}?api_key={API_KEY}&language=en-US"
    collection_data = requests.get(collection_url).json()
    return collection_data.get('parts', [])

@st.cache_resource(show_spinner="Finding movies that are part of a franchise...")
def get_franchise_movie_options():
    franchise_movie_list = []
    for title in movie_names[:500]:
        movie_record = movies[movies['title'] == title]
        if not movie_record.empty:
            movie_id = movie_record['id'].values[0]
            movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
            try:
                movie_data = requests.get(movie_details_url).json()
                if movie_data.get('belongs_to_collection'):
                    franchise_movie_list.append(title)
            except requests.exceptions.RequestException:
                continue
    return ["Select a franchise movie"] + sorted(franchise_movie_list)

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
    details = fetch_details(movie_id)
    details['poster'] = fetch_poster(movie_id)
    details['trailer'] = fetch_trailer(movie_id)
    return details

def display_movie_details(movie_id, details=None):
    if not details: details = fetch_details(movie_id)
    trailer = fetch_trailer(movie_id)
    
    st.markdown(f"**Rating:** {details.get('rating', 0):.1f}/10")
    st.markdown(f"**Released:** {details.get('release_date', 'N/A')}")
    st.markdown(f"**Runtime:** {details.get('runtime', 0)} mins")
    st.markdown(f"**Genres:** {details.get('genres', 'N/A')}")
    st.write(details.get('overview', 'No overview available.'))
    if details.get('imdb_link'): st.markdown(f"[IMDb Page]({details['imdb_link']})", unsafe_allow_html=True)
    if trailer: st.video(trailer)

def get_movie_title(movie_id):
    movie_record = movies[movies['id'] == movie_id]
    if not movie_record.empty: return movie_record['title'].values[0]
    else: return fetch_details(movie_id).get('title', 'Title Not Found')

def initialize_session_state():
    keys = ['selected_movie', 'franchise_movie', 'mood_surprise', 'mood_history', 'genre_surprise_movie']
    defaults = {
        'selected_movie': "Select a movie",
        'franchise_movie': "Select a franchise movie",
        'mood_surprise': None,
        'mood_history': [],
        'genre_surprise_movie': None
    }
    for key in keys:
        if key not in st.session_state:
            st.session_state[key] = defaults.get(key)

initialize_session_state()


st.sidebar.header("🎬 Movie Recommender")

def go_home():
    st.session_state.selected_movie = "Select a movie"
    st.session_state.mood_surprise = None
    st.session_state.genre_surprise_movie = None
    st.session_state.franchise_movie = "Select a franchise movie"

def surprise_me():
    """Selects a random movie and sets it as the selected movie."""
    valid_movie_names = [name for name in movie_names if name != "Select a movie"]
    if valid_movie_names:
        random_movie = random.choice(valid_movie_names)
        st.session_state.selected_movie = random_movie
        st.session_state.mood_surprise = None
        st.session_state.genre_surprise_movie = None
        st.session_state.franchise_movie = "Select a franchise movie"

st.sidebar.button("🏠 Home", on_click=go_home, use_container_width=True)
st.sidebar.button("✨ Surprise Me", on_click=surprise_me, use_container_width=True) # <-- NEW BUTTON
st.sidebar.markdown("---")

st.sidebar.header("🔍 Find by Title")

movie_names_with_placeholder = ["Select a movie"] + list(movie_names)
selected_movie = st.sidebar.selectbox("Select to get recommendations", movie_names_with_placeholder, key="selected_movie")

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Recommendation Filters")
rating_options = [x / 2 for x in range(0, 21)]
min_rating = st.sidebar.selectbox("Minimum Rating", rating_options, index=10)
year_range = st.sidebar.slider("Release Year Range", 1950, 2025, (1990, 2025))

# --- Main Page ---
if selected_movie != "Select a movie":
    st.title(f"Recommendations for: *{selected_movie}*")
    with st.spinner("Finding your perfect movies... 🍿"):
        recommended_movie_ids = recommend(selected_movie)
        
        filtered_recs = []
        for movie_id in recommended_movie_ids:
            details = fetch_details(movie_id)
            if details.get('rating', 0) >= min_rating:
                date = details.get('release_date', '')
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
            st.error("No movies found with your filters. Try adjusting them!")
else:
    st.title("🎬 Movie Recomendatio Engine")
    st.markdown("Explore movies in new ways, Choose an option below to get started.")

    with st.expander("✨ 1. Surprise Me With A Mood", expanded=True):
        mood_genre_map = {
            "Happy & Uplifting": [35, 10751],
            "Romantic & Heartfelt": [10749, 18],
            "Nostalgic & Bittersweet": [18, 10749],
            "Underdog & Inspirational": [18, 36],
            "Action-Packed Adventure": [28, 12],
            "Gritty & Intense": [80, 53],
            "Scared & On-Edge": [27, 9648],
            "Epic & Historical": [36, 10752, 12],
            "Mind-Bending & Mysterious": [9648, 878],
            "Imaginative & Other-Worldly": [14, 878, 16],
            "Laugh-Out-Loud Comedy": [35],
        }
        selected_mood = st.selectbox("How are you feeling today?", list(mood_genre_map.keys()))

        def find_mood_movie():
            genre_ids = ",".join(map(str, mood_genre_map[selected_mood]))
            url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&page=1&primary_release_date.gte=1990-01-01&vote_count.gte=200&with_genres={genre_ids}"
            try:
                data = requests.get(url).json()
                potential = [m for m in data.get('results', []) if isinstance(m, dict) and m.get('id') not in st.session_state.mood_history]
                if len(potential) < 3:
                    st.toast("Found few new movies. Resetting history for more options.", icon="🔄")
                    st.session_state.mood_history = []
                    potential.extend([m for m in data.get('results', []) if isinstance(m, dict) and m.get('id') not in [p.get('id') for p in potential]])
                
                if potential:
                    num_to_suggest = min(len(potential), 3)
                    chosen_movies = random.sample(potential, k=num_to_suggest)
                    st.session_state.mood_surprise = chosen_movies
                    for movie in chosen_movies: st.session_state.mood_history.append(movie.get('id'))
                else: 
                    st.toast("Couldn't find any movies for that mood.", icon="😞")
                    st.session_state.mood_surprise = None
            except Exception as e: st.error(f"API Error: {e}")

        st.button("Get Mood Suggestions", on_click=find_mood_movie, use_container_width=True)
        if st.session_state.mood_surprise:
            st.markdown(f"### For your mood, we suggest one of these:")
            movies_to_show = st.session_state.mood_surprise
            if isinstance(movies_to_show, list):
                cols = st.columns(len(movies_to_show))
                for i, movie in enumerate(movies_to_show):
                    with cols[i]:
                        if isinstance(movie, dict):
                            st.image(fetch_poster(movie.get('id')), use_container_width=True)
                            st.markdown(f"**{movie.get('title', 'No Title')}**")
                            with st.expander("Details"): display_movie_details(movie.get('id'))

    with st.expander("🆚 2. Compare Movies"):
        col1, col2 = st.columns(2)
        with col1: movie1_title = st.selectbox("Select Movie 1", movie_names_with_placeholder, key="movie1")
        with col2: movie2_title = st.selectbox("Select Movie 2", movie_names_with_placeholder, key="movie2")

        if movie1_title != "Select a movie" and movie2_title != "Select a movie":
            movie1_details = get_movie_for_compare(movie1_title)
            movie2_details = get_movie_for_compare(movie2_title)

            if movie1_details and movie2_details:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<div class='compare-card'>", unsafe_allow_html=True)
                    st.image(movie1_details['poster'])
                    st.subheader(movie1_details['title'])
                    st.write(f"**⭐ Rating:** {movie1_details.get('rating', 0):.1f}/10")
                    st.write(f"**🗓️ Release:** {movie1_details.get('release_date', 'N/A')}")
                    st.write(f"**⏳ Runtime:** {movie1_details.get('runtime', 0)} mins")
                    st.write(f"**Genres:** {movie1_details.get('genres', 'N/A')}")
                    if movie1_details.get('trailer'): st.markdown(f"[Watch Trailer]({movie1_details['trailer']})")
                    st.markdown("</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='compare-card'>", unsafe_allow_html=True)
                    st.image(movie2_details['poster'])
                    st.subheader(movie2_details['title'])
                    st.write(f"**⭐ Rating:** {movie2_details.get('rating', 0):.1f}/10")
                    st.write(f"**🗓️ Release:** {movie2_details.get('release_date', 'N/A')}")
                    st.write(f"**⏳ Runtime:** {movie2_details.get('runtime', 0)} mins")
                    st.write(f"**Genres:** {movie2_details.get('genres', 'N/A')}")
                    if movie2_details.get('trailer'): st.markdown(f"[Watch Trailer]({movie2_details['trailer']})")
                    st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("🎭 3. Discover by Multiple Genres"):
        genre_map = fetch_genres()
        if genre_map:
            selected_genres = st.multiselect("Select genres to find movies containing ALL of them", list(genre_map.keys()))
            
            def find_movie_by_genres():
                if not selected_genres: st.toast("Please select at least one genre.", icon="⚠️"); return
                genre_ids = ",".join([str(genre_map[name]) for name in selected_genres])
                url = f"https://api.themoviedb.org/3/discover/movie?api_key={API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&page=1&primary_release_date.gte=1990-01-01&vote_count.gte=200&with_genres={genre_ids}"
                try:
                    data = requests.get(url).json()
                    results = [m for m in data.get('results', []) if isinstance(m, dict)]
                    if results: 
                        num_to_suggest = min(len(results), 3)
                        st.session_state.genre_surprise_movie = random.sample(results, k=num_to_suggest)
                    else: 
                        st.toast("No movies found with that exact genre combination.", icon="😞")
                        st.session_state.genre_surprise_movie = None
                except Exception as e: st.error(f"API Error: {e}")
            
            st.button("Find by Genres", on_click=find_movie_by_genres, use_container_width=True)
            if st.session_state.genre_surprise_movie:
                st.markdown(f"### Popular movies with your selected genres:")
                movies_to_show = st.session_state.genre_surprise_movie
                if isinstance(movies_to_show, list):
                    cols = st.columns(len(movies_to_show))
                    for i, movie in enumerate(movies_to_show):
                        with cols[i]:
                            if isinstance(movie, dict):
                                st.image(fetch_poster(movie.get('id')), use_container_width=True)
                                st.markdown(f"**{movie.get('title', 'No Title')}**")
                                with st.expander("Details"): display_movie_details(movie.get('id'))

    with st.expander("🎬 4. Watch Movie Franchises"):
        franchise_options = get_franchise_movie_options()
        selected_franchise_movie = st.selectbox("Select a franchise movie to see its collection", franchise_options, key="franchise_movie")
        
        if selected_franchise_movie != "Select a franchise movie":
            movie_record = movies[movies['title'] == selected_franchise_movie]
            if not movie_record.empty:
                movie_id = movie_record['id'].values[0]
                with st.spinner("Fetching franchise details..."):
                    collection_parts = fetch_collection_details(movie_id)
                if collection_parts:
                    st.subheader(f"Movies in the '{selected_franchise_movie}' Franchise:")
                    parts_to_show = [p for p in collection_parts if isinstance(p, dict)]
                    cols = st.columns(min(len(parts_to_show), 6))
                    for i, part in enumerate(parts_to_show):
                        with cols[i % 6]:
                            st.image(fetch_poster(part.get('id')), use_container_width=True)
                            st.markdown(f"**{part.get('title', 'No Title')}**")
                            with st.expander("Details"): display_movie_details(part.get('id'))

    st.markdown("---")
    st.markdown("## 🔥 Trending This Week")
    trending_movies_response = requests.get(f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1").json()
    cols = st.columns(5)
    trending_results = [m for m in trending_movies_response.get('results', []) if isinstance(m, dict)]
    for i, movie in enumerate(trending_results[:5]):
        with cols[i]:
            st.image(fetch_poster(movie.get('id')), use_container_width=True)
            st.markdown(f"**{movie.get('title', 'No Title')}**")
            with st.expander("Details"): display_movie_details(movie.get('id'))

    st.markdown("---")
    st.markdown("## 🏆 Top Rated Movies")
    top_rated_response = requests.get(f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1").json()
    cols = st.columns(5)
    top_rated_results = [m for m in top_rated_response.get('results', []) if isinstance(m, dict)]
    for i, movie in enumerate(top_rated_results[:5]):
        with cols[i]:
            st.image(fetch_poster(movie.get('id')), use_container_width=True)
            st.markdown(f"**{movie.get('title', 'No Title')}**")
            with st.expander("Details"): display_movie_details(movie.get('id'))
