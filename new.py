import streamlit as st
import pickle
import requests
from sklearn.neighbors import NearestNeighbors
import difflib

st.set_page_config(page_title="🎬 Movie Recommender 🎬", layout="wide")

st.markdown("""
<style>
    body { background-color: #121212; color: white; }
    .movie-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 10px;
        border-radius: 15px;
        transition: 0.3s;
        margin-bottom: 20px;
        text-align: center;
    }
    .movie-card:hover {
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(255,255,255,0.2);
    }
    .stButton>button {
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #ff7777;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Movie Recommender 🎬")

movies = pickle.load(open("movies_data.pkl", "rb"))
tfidf_matrix = pickle.load(open("tfidf_matrix.pkl", "rb"))
movie_names = movies['title'].values

API_KEY = "bb8c8e12742c72ae502a3863ccb5402a"

@st.cache_data
def fetch_poster(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    poster_path = data.get('poster_path')
    return "https://image.tmdb.org/t/p/w500/" + poster_path if poster_path else "https://via.placeholder.com/500x750?text=No+Image"

@st.cache_data
def fetch_details(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    overview = data.get('overview', 'No description available.')
    release_date = data.get('release_date', 'Unknown')
    rating = data.get('vote_average', 0)
    genres = ', '.join([g['name'] for g in data.get('genres', [])])
    runtime = data.get('runtime', 'Unknown')
    imdb_id = data.get('imdb_id')
    imdb_link = f"https://www.imdb.com/title/{imdb_id}/" if imdb_id else None
    return overview, release_date, rating, genres, runtime, imdb_link

@st.cache_data
def fetch_trailer(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}&language=en-US"
    data = requests.get(url).json()
    for video in data.get('results', []):
        if video['type'] == 'Trailer' and video['site'] == 'YouTube':
            return f"https://www.youtube.com/watch?v={video['key']}"
    return None

@st.cache_data
def fetch_trending_movies():
    url = f"https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1"
    data = requests.get(url).json()
    trending_movies = []
    for m in data['results'][:20]:
        poster = "https://image.tmdb.org/t/p/w500/" + m['poster_path'] if m.get('poster_path') else "https://via.placeholder.com/500x750?text=No+Image"
        movie_id = m['id']
        trending_movies.append((m['title'], poster, movie_id))
    return trending_movies

@st.cache_data
def fetch_top_movies():
    url = f"https://api.themoviedb.org/3/movie/top_rated?api_key={API_KEY}&language=en-US&page=1"
    data = requests.get(url).json()
    top_movies = []
    for m in data['results'][:20]:
        poster = "https://image.tmdb.org/t/p/w500/" + m['poster_path'] if m.get('poster_path') else "https://via.placeholder.com/500x750?text=No+Image"
        movie_id = m['id']
        top_movies.append((m['title'], poster, movie_id))
    return top_movies

# Model for recommendations
nbrs = NearestNeighbors(n_neighbors=6, metric='cosine').fit(tfidf_matrix)

def recommend(title):
    matches = difflib.get_close_matches(title.lower(), movies['title'].str.lower(), n=1, cutoff=0.6)
    if not matches:
        return []
    idx = movies[movies['title'].str.lower() == matches[0]].index[0]
    distances, indices = nbrs.kneighbors(tfidf_matrix[idx])
    recommendations = []
    for i in indices[0][1:]:
        movie_id = movies.iloc[i].id
        title = movies.iloc[i].title
        poster = fetch_poster(movie_id)
        overview, date, rating, genres, runtime, imdb_link = fetch_details(movie_id)
        trailer = fetch_trailer(movie_id)
        recommendations.append({
            "id": movie_id,
            "title": title,
            "poster": poster,
            "overview": overview,
            "date": date,
            "rating": rating,
            "genres": genres,
            "runtime": runtime,
            "trailer": trailer,
            "imdb": imdb_link
        })
    return recommendations

st.markdown("## 🎯 Movie Recommender")
selected_movie = st.selectbox("🎯 Type or select a movie", movie_names)

# ✅ Both selectboxes
rating_options = [x / 2 for x in range(0, 21)]  # 0.0 to 10.0 step 0.5
min_rating = st.selectbox("⭐ Minimum Rating", rating_options, index=0)

year_range = st.slider("📅 Release Year Range", 1950, 2025, (1990, 2025))

if st.button("Show Recommendations"):
    with st.spinner("Finding your perfect movies... 🍿"):
        recommendations = recommend(selected_movie)
        recommendations = [
            r for r in recommendations
            if r['rating'] >= min_rating and r['date'] and r['date'][:4].isdigit()
            and year_range[0] <= int(r['date'][:4]) <= year_range[1]
        ]
        if recommendations:
            cols = st.columns(5)
            for i, rec in enumerate(recommendations):
                with cols[i % 5]:
                    st.markdown(f"<div class='movie-card'>", unsafe_allow_html=True)
                    st.image(rec['poster'], use_container_width=True)
                    st.markdown(f"### {rec['title']}")
                    st.markdown(f"**Release:** {rec['date']}  \n**⭐ Rating:** {rec['rating']}/10  \n**Genres:** {rec['genres']}  \n**Runtime:** {rec['runtime']} mins")
                    st.write(rec['overview'][:150] + "...")
                    if rec['imdb']:
                        st.markdown(f"[IMDb Link]({rec['imdb']})")
                    if rec['trailer']:
                        st.video(rec['trailer'])
                    st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("No movies found based on your filters. Try adjusting them!")

# 🔥 Trending Movies
chunk_size = 5
st.markdown("## 🔥 Trending This Week")
trending_movies = fetch_trending_movies()
for start in range(0, len(trending_movies), chunk_size):
    cols = st.columns(chunk_size)
    for i, col in enumerate(cols):
        if start + i < len(trending_movies):
            title, poster, movie_id = trending_movies[start + i]
            with col:
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title}**")
                with st.expander(f"🎬 Details for {title}"):
                    overview, date, rating, genres, runtime, imdb_link = fetch_details(movie_id)
                    trailer = fetch_trailer(movie_id)
                    st.write(f"**Release:** {date}")
                    st.write(f"**Rating:** {rating}/10")
                    st.write(f"**Genres:** {genres}")
                    st.write(f"**Runtime:** {runtime} mins")
                    st.write(overview)
                    if imdb_link:
                        st.markdown(f"[IMDb Link]({imdb_link})")
                    if trailer:
                        st.video(trailer)

# 🏆 Top Rated Movies
st.markdown("## 🏆 Top Rated Movies")
top_movies = fetch_top_movies()
for start in range(0, len(top_movies), chunk_size):
    cols = st.columns(chunk_size)
    for i, col in enumerate(cols):
        if start + i < len(top_movies):
            title, poster, movie_id = top_movies[start + i]
            with col:
                st.image(poster, use_container_width=True)
                st.markdown(f"**{title}**")
                with st.expander(f"🎬 Details for {title}"):
                    overview, date, rating, genres, runtime, imdb_link = fetch_details(movie_id)
                    trailer = fetch_trailer(movie_id)
                    st.write(f"**Release:** {date}")
                    st.write(f"**Rating:** {rating}/10")
                    st.write(f"**Genres:** {genres}")
                    st.write(f"**Runtime:** {runtime} mins")
                    st.write(overview)
                    if imdb_link:
                        st.markdown(f"[IMDb Link]({imdb_link})")
                    if trailer:
                        st.video(trailer)
