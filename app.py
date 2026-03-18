"""
app.py — CineScope Movie Recommender
─────────────────────────────────────
• On first run (or whenever .pkl files are missing) the app automatically
  runs model_builder.build_pipeline() with a live progress log — no user
  action required beyond having Dataset/movie_dataset.csv in place.

• Three recommendation pages unlock once the models are ready:
    🎯  Recommend Me   — content-based top-10 similar movies
    🎞️  Movie Details  — full metadata card
    🔥  Top Movies     — filterable ranked list
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="CineScope",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Hide the sidebar toggle arrow — no sidebar used */
[data-testid="collapsedControl"] { display: none !important; }
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;0,900;1,400;1,700&family=Barlow:wght@300;400;500;600&family=Barlow+Condensed:wght@400;600;700&display=swap');

:root {
    --bg:      #0a0a0b;
    --surface: #111114;
    --card:    #18181c;
    --border:  #2a2a30;
    --gold:    #c9a84c;
    --gold-lt: #e8c96a;
    --text:    #e8e4dc;
    --muted:   #6b6b75;
    --radius:  10px;
}

html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: var(--bg);
    color: var(--text);
}

/* ── Brand ── */
.brand-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-style: italic;
    font-weight: 900;
    background: linear-gradient(100deg, #c9a84c 0%, #e8c96a 50%, #a07830 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0; padding: 0; line-height: 1.1;
}
.brand-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.26em;
    text-transform: uppercase;
    color: var(--muted);
}

/* ── Page heading ── */
.pg-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.85rem;
    font-style: italic;
    color: var(--text);
    margin-bottom: 0.1rem;
}
.pg-sub {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 1.5rem;
}

/* ── Movie card ── */
.movie-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    transition: transform .22s ease, border-color .22s ease, box-shadow .22s ease;
}
.movie-card:hover {
    transform: translateY(-5px);
    border-color: var(--gold);
    box-shadow: 0 14px 40px rgba(201,168,76,.18);
}
.movie-card img {
    width: 100%;
    display: block;
    aspect-ratio: 2/3;
    object-fit: cover;
}
.movie-card-body { padding: .6rem .75rem .75rem; }
.movie-card-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-weight: 700;
    font-size: .88rem;
    line-height: 1.3;
    color: var(--text);
    margin-bottom: .18rem;
}
.movie-card-meta { font-size: .7rem; color: var(--muted); }
.star { color: var(--gold); }

/* ── Detail page ── */
.detail-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.5rem;
    font-weight: 900;
    line-height: 1.15;
    margin-bottom: .3rem;
}
.badge {
    display: inline-block;
    padding: .16rem .6rem;
    border-radius: 999px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .7rem;
    font-weight: 700;
    letter-spacing: .07em;
    text-transform: uppercase;
    margin-right: .3rem;
    margin-bottom: .3rem;
    border: 1px solid rgba(201,168,76,.35);
    color: var(--gold);
    background: rgba(201,168,76,.08);
}
.meta-row { font-size: .85rem; color: var(--muted); margin-bottom: .35rem; }
.meta-row strong { color: var(--text); }
.overview-block {
    font-size: .93rem;
    line-height: 1.8;
    color: #c0bcb4;
    border-left: 3px solid var(--gold);
    padding-left: 1rem;
    margin: 1rem 0 1.4rem;
}
.crew-lbl {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: .18rem;
}
.crew-val { font-size: .86rem; color: var(--text); }

/* ── Build / log box ── */
.log-box {
    background: #060608;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Courier New', monospace;
    font-size: .78rem;
    color: #7ef0a0;
    min-height: 200px;
    max-height: 340px;
    overflow-y: auto;
    white-space: pre-wrap;
    line-height: 1.65;
}
.build-info {
    background: var(--card);
    border: 1px solid var(--border);
    border-left: 3px solid var(--gold);
    border-radius: 8px;
    padding: .9rem 1.1rem;
    font-size: .88rem;
    margin-bottom: 1rem;
}

/* ── Stat cards ── */
.stat-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: .9rem 1rem;
    text-align: center;
}
.stat-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--gold);
}
.stat-lbl {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .66rem;
    font-weight: 700;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-top: .12rem;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #b8920e, #c9a84c);
    color: #0a0a0b;
    border: none;
    border-radius: 6px;
    font-family: 'Barlow Condensed', sans-serif;
    font-size: .82rem;
    font-weight: 700;
    letter-spacing: .1em;
    text-transform: uppercase;
    padding: .5rem 1.4rem;
    transition: all .2s;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #c9a84c, #e8c96a);
    box-shadow: 0 4px 20px rgba(201,168,76,.4);
}

/* ── Inputs ── */
.stSelectbox > div > div { background: var(--card) !important; border-color: var(--border) !important; }
.stMultiSelect > div > div { background: var(--card) !important; border-color: var(--border) !important; }
.stSlider > div > div > div { background: var(--gold) !important; }
.stProgress > div > div > div { background: var(--gold) !important; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
hr { border-color: var(--border); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PKL_FILES   = ["df_popular.pkl", "df_recommend.pkl", "similarity.pkl"]
TMDB_BASE   = "https://image.tmdb.org/t/p/w500/"
PLACEHOLDER = "https://placehold.co/500x750/18181c/6b6b75?text=No+Poster"


def pkls_exist() -> bool:
    return all(Path(f).exists() for f in PKL_FILES)


def poster_url(path) -> str:
    if not path or (isinstance(path, float)):
        return PLACEHOLDER
    s = str(path).strip()
    return (TMDB_BASE + s) if s else PLACEHOLDER


def fmt_list(val) -> str:
    if isinstance(val, list):
        items = [str(v).strip() for v in val if str(v).strip()]
        return ", ".join(items) or "—"
    return str(val) if val else "—"


def movie_card_html(title: str, poster: str, rating=None, year=None) -> str:
    meta_parts = []
    if rating:
        meta_parts.append(f'<span class="star">★</span> {float(rating):.1f}')
    if year:
        meta_parts.append(str(year))
    meta = " &nbsp;·&nbsp; ".join(meta_parts)
    return (
        f'<div class="movie-card">'
        f'<img src="{poster}" alt="{title}" loading="lazy" '
        f'onerror="this.src=\'{PLACEHOLDER}\'">'
        f'<div class="movie-card-body">'
        f'<div class="movie-card-title">{title}</div>'
        f'<div class="movie-card-meta">{meta}</div>'
        f'</div></div>'
    )


def page_header(icon: str, title: str, sub: str):
    st.markdown(
        f'<div style="padding:1.6rem 0 .3rem">'
        f'<div class="pg-title">{icon} {title}</div>'
        f'<div class="pg-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Auto-build: run model_builder if .pkl files are missing
# ─────────────────────────────────────────────────────────────────────────────

if not pkls_exist():
    st.markdown(
        '<div style="padding:1.8rem 0 .5rem">'
        '<div class="brand-title">CineScope</div>'
        '<div class="brand-sub">Movie Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<hr style="margin:.2rem 0 1.2rem"/>', unsafe_allow_html=True)

    st.markdown(
        '<div class="build-info">'
        '🔧 <strong>First-time setup detected</strong> — no model files found.<br>'
        'Building the recommendation engine from '
        '<code>Dataset/movie_dataset.csv</code> now. This runs once and '
        'typically takes 30–120 seconds depending on dataset size.'
        '</div>',
        unsafe_allow_html=True,
    )

    log_box   = st.empty()
    prog_bar  = st.progress(0, text="Initialising…")
    stat_area = st.empty()

    log_lines: list[str] = []

    def append_log(msg: str):
        log_lines.append(msg)
        log_box.markdown(
            f'<div class="log-box">{"<br>".join(log_lines)}</div>',
            unsafe_allow_html=True,
        )

    # Map log keywords → progress percentage
    STEPS = [
        ("Loading dataset",          8),
        ("Dropping rows",           18),
        ("Building df_popular",      32),
        ("Building df_recommend",    50),
        ("TF-IDF",                   68),
        ("cosine similarity",        84),
        ("Saving pickle",            96),
        ("Pipeline complete",       100),
    ]

    def log_with_progress(msg: str):
        append_log(msg)
        for kw, pct in STEPS:
            if kw.lower() in msg.lower():
                prog_bar.progress(pct, text=msg[:70])
                break

    try:
        from model_builder import build_pipeline
        stats = build_pipeline(log=log_with_progress)

        prog_bar.progress(100, text="✅ Done!")

        # Stat cards
        cols = stat_area.columns(4)
        for col, val, lbl in [
            (cols[0], f"{stats['n_raw']:,}",       "Raw rows"),
            (cols[1], f"{stats['n_clean']:,}",      "After dropna"),
            (cols[2], f"{stats['n_recommend']:,}",  "Indexed movies"),
            (cols[3], f"{stats['n_popular']:,}",    "Popular movies"),
        ]:
            col.markdown(
                f'<div class="stat-card">'
                f'<div class="stat-val">{val}</div>'
                f'<div class="stat-lbl">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        st.success("✅ Model built successfully! Reloading the app…")
        st.rerun()

    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Make sure `Dataset/movie_dataset.csv` exists in the same directory "
            "as `app.py`, then refresh the page."
        )
        st.stop()

    except Exception as e:
        st.error(f"Build failed: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# Load pickles (cached — reloads only when files change)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_data():
    df_popular   = pd.DataFrame(pickle.load(open("df_popular.pkl",   "rb")))
    df_recommend = pd.DataFrame(pickle.load(open("df_recommend.pkl", "rb")))
    similarity   = pickle.load(open("similarity.pkl", "rb"))

    df_popular["release_date"] = pd.to_datetime(
        df_popular["release_date"], errors="coerce"
    )

    # Restore list columns (serialised as dicts by to_dict())
    for col in ["genres", "production_countries"]:
        df_popular[col] = df_popular[col].apply(
            lambda x: list(x.values()) if isinstance(x, dict) else
                      (x if isinstance(x, list) else [])
        )

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
                    st.image(poster, use_column_width=True)
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
            st.image(fetch_poster(movie_data['poster_path']), use_column_width=True)
            
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

        # Genre filter
        if sel_genre != "All Genres":
            f = f[f["genres"].apply(
                lambda x: sel_genre in x if isinstance(x, list) else False
            )]

        # Country filter
        if sel_country != "All Countries":
            f = f[f["production_countries"].apply(
                lambda x: sel_country in x if isinstance(x, list) else False
            )]

        # Quality floor
        f = f[
            (f["numVotes"]   >= 5000) &
            (f["popularity"] >= 10)
        ]

        return f.sort_values(
            by=["averageRating", "numVotes"], ascending=False
        ).head(40)

    results = apply_filters(df_popular)

    # ── Render grid ───────────────────────────────────────────────────────
    if results.empty:
        st.warning("No movies match the selected filters — try a different combination.")
    else:
        # Active filter chips
        active = []
        if sel_genre    != "All Genres":    active.append(f"🎭 {sel_genre}")
        if sel_country  != "All Countries": active.append(f"🌍 {sel_country}")
        if sel_year_label    != "All Years":     active.append(f"📅 {sel_year_label}")
        if sel_runtime_label != "Any Length":    active.append(f"⏱️ {sel_runtime_label}")

        chips_html = "".join(
            f'<span class="badge" style="margin-right:.4rem">{c}</span>'
            for c in active
        ) if active else ""

        st.markdown(
            f'<div style="display:flex;align-items:center;gap:.8rem;margin-bottom:1rem">'
            f'<span style="font-size:.8rem;color:var(--muted)">'
            f'Showing <strong style="color:var(--gold)">{len(results)}</strong> movies</span>'
            f'{"&nbsp;" + chips_html if chips_html else ""}'
            f'</div>',
            unsafe_allow_html=True,
        )

        NUM_COLS = 4
        for i in range(0, len(results), NUM_COLS):
            cols = st.columns(NUM_COLS, gap="small")
            for j in range(NUM_COLS):
                if i + j < len(results):
                    mv   = results.iloc[i + j]
                    year = mv["release_date"].year if pd.notna(mv["release_date"]) else None
                    with cols[j]:
                        st.image(fetch_poster(movie['poster_path']), use_column_width=True)
                        st.markdown(f"**{movie['title']}**")
                        st.caption(f"⭐ {movie['averageRating']} | 🗓️ {movie['release_date'].year}")