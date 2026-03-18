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


# ─────────────────────────────────────────────────────────────────────────────
# Top bar
# ─────────────────────────────────────────────────────────────────────────────

brand_col, nav_col = st.columns([1, 3])

with brand_col:
    st.markdown(
        '<div style="padding:.9rem 0 .4rem">'
        '<div class="brand-title">CineScope</div>'
        '<div class="brand-sub">Movie Intelligence</div>'
        '</div>',
        unsafe_allow_html=True,
    )

with nav_col:
    selected = option_menu(
        menu_title=None,
        options=["Recommend Me", "Movie Details", "Top Movies"],
        icons=["stars", "camera-reels-fill", "fire"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {
                "padding": ".6rem 0",
                "background": "transparent",
            },
            "nav-link": {
                "font-family": "'Barlow Condensed', sans-serif",
                "font-size":   ".82rem",
                "font-weight": "700",
                "letter-spacing": ".12em",
                "text-transform": "uppercase",
                "color":       "#6b6b75",
                "padding":     ".52rem 1.1rem",
                "border-radius": "6px",
            },
            "nav-link-selected": {
                "background":  "rgba(201,168,76,.12)",
                "color":       "#c9a84c",
                "border":      "1px solid rgba(201,168,76,.3)",
            },
            "icon": {"font-size": ".88rem"},
        },
    )

st.markdown('<hr style="margin:0 0 .4rem"/>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Recommend Me
# ─────────────────────────────────────────────────────────────────────────────

if selected == "Recommend Me":
    page_header("🎯", "Recommend Me a Movie", "Content-based top-10 similar films")

    chosen = st.selectbox(
        "Pick a movie you love",
        df_recommend["title"].values,
        label_visibility="collapsed",
        placeholder="Type or select a movie…",
    )

    if st.button("Find Similar Movies  →"):
        idx       = df_recommend[df_recommend["title"] == chosen].index[0]
        distances = similarity[idx]
        top10     = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:11]

        st.markdown(
            f'<p style="font-size:.82rem;color:var(--muted);margin:.4rem 0 1.1rem">'
            f'Because you liked '
            f'<strong style="color:var(--gold)">{chosen}</strong></p>',
            unsafe_allow_html=True,
        )

        for row_start in range(0, 10, 5):
            cols = st.columns(5, gap="small")
            for ci, (midx, _score) in enumerate(top10[row_start:row_start + 5]):
                row = df_recommend.iloc[midx]
                try:
                    year = pd.to_datetime(row.get("release_date", "")).year
                except Exception:
                    year = None
                with cols[ci]:
                    st.markdown(
                        movie_card_html(
                            row["title"],
                            poster_url(row.get("poster_path")),
                            rating=row.get("averageRating"),
                            year=year,
                        ),
                        unsafe_allow_html=True,
                    )
            st.markdown("<div style='margin-bottom:.7rem'/>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Movie Details
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Movie Details":
    page_header("🎞️", "Movie Details", "Full metadata, crew and overview")

    chosen = st.selectbox(
        "Select a movie",
        df_recommend["title"].values,
        label_visibility="collapsed",
    )

    if chosen:
        rec = df_recommend[df_recommend["title"] == chosen].iloc[0]
        pop = df_popular[df_popular["title"] == chosen]

        col_poster, col_info = st.columns([1, 2], gap="large")

        with col_poster:
            st.image(poster_url(rec.get("poster_path")), use_container_width=True)

        with col_info:
            st.markdown(
                f'<div class="detail-title">{rec["title"]}</div>',
                unsafe_allow_html=True,
            )

            # Genre badges
            genres_raw = (
                pop["genres"].iloc[0]
                if not pop.empty and isinstance(pop["genres"].iloc[0], list)
                else []
            )
            if genres_raw:
                st.markdown(
                    "".join(f'<span class="badge">{g}</span>' for g in genres_raw[:7]),
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='margin:.5rem 0'/>", unsafe_allow_html=True)

            def meta_row(label: str, value: str):
                st.markdown(
                    f'<div class="meta-row"><strong>{label}</strong> {value}</div>',
                    unsafe_allow_html=True,
                )

            # Release date
            try:
                rd = pd.to_datetime(rec.get("release_date", ""))
                meta_row("📅 Release", rd.strftime("%B %d, %Y"))
            except Exception:
                meta_row("📅 Release", "N/A")

            meta_row("⏱️ Runtime", f"{rec.get('runtime', '—')} min")

            rating = rec.get("averageRating")
            if rating:
                r = float(rating)
                stars = "★" * round(r / 2)
                meta_row(
                    "⭐ Rating",
                    f'<span style="color:var(--gold)">{r:.1f}/10 &nbsp;{stars}</span>',
                )

            # Overview
            st.markdown(
                f'<div class="overview-block">'
                f'{rec.get("overview", "No overview available.")}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Crew
            def clean_names(raw) -> str:
                if not isinstance(raw, list):
                    return "—"
                return ", ".join(
                    n.strip() for n in raw if str(n).strip()
                ) or "—"

            c1, c2, c3 = st.columns(3, gap="medium")
            for col, lbl, val in [
                (c1, "🎬 Director", clean_names(rec.get("directors", [])[:2])),
                (c2, "✍️ Writer",   clean_names(rec.get("writers",   [])[:2])),
                (c3, "🎭 Top Cast", clean_names(rec.get("cast",      [])[:5])),
            ]:
                with col:
                    st.markdown(
                        f'<div class="crew-lbl">{lbl}</div>'
                        f'<div class="crew-val">{val}</div>',
                        unsafe_allow_html=True,
                    )

            # Countries
            if not pop.empty:
                countries = pop["production_countries"].iloc[0]
                if isinstance(countries, list) and countries:
                    st.markdown("<div style='margin-top:.9rem'/>", unsafe_allow_html=True)
                    st.markdown(
                        f'<div class="crew-lbl">🌍 Production Countries</div>'
                        f'<div class="crew-val">{", ".join(countries)}</div>',
                        unsafe_allow_html=True,
                    )


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Top Movies
# ─────────────────────────────────────────────────────────────────────────────

elif selected == "Top Movies":
    page_header("🔥", "Top Popular Movies", "Filter by genre · country · year · runtime")

    # ── Build option lists from data ──────────────────────────────────────
    all_genres = ["All Genres"] + sorted({
        g for sub in df_popular["genres"]
        if isinstance(sub, list)
        for g in sub if g
    })
    all_countries = ["All Countries"] + sorted({
        c for sub in df_popular["production_countries"]
        if isinstance(sub, list)
        for c in sub if c
    })

    YEAR_OPTIONS = {
        "All Years":   (0,    9999),
        "2020s":       (2020, 9999),
        "2010s":       (2010, 2019),
        "2000s":       (2000, 2009),
        "1990s":       (1990, 1999),
        "1980s":       (1980, 1989),
        "Before 1980": (0,    1979),
    }

    RUNTIME_OPTIONS = {
        "Any Length":          (0,   9999),
        "Short  (< 90 min)":   (0,   89),
        "Standard  (90–120)":  (90,  120),
        "Long  (120–150 min)": (121, 150),
        "Epic  (> 150 min)":   (151, 9999),
    }

    # ── Inline filter row ─────────────────────────────────────────────────
    st.markdown(
        '<div style="background:var(--card);border:1px solid var(--border);'
        'border-radius:10px;padding:1.1rem 1.3rem 0.9rem;margin-bottom:1.4rem">',
        unsafe_allow_html=True,
    )

    fc1, fc2, fc3, fc4 = st.columns(4, gap="medium")

    with fc1:
        st.markdown(
            '<div class="crew-lbl" style="margin-bottom:.35rem">🎭 Genre</div>',
            unsafe_allow_html=True,
        )
        sel_genre = st.selectbox(
            "Genre", all_genres,
            label_visibility="collapsed",
            key="filter_genre",
        )

    with fc2:
        st.markdown(
            '<div class="crew-lbl" style="margin-bottom:.35rem">🌍 Country</div>',
            unsafe_allow_html=True,
        )
        sel_country = st.selectbox(
            "Country", all_countries,
            label_visibility="collapsed",
            key="filter_country",
        )

    with fc3:
        st.markdown(
            '<div class="crew-lbl" style="margin-bottom:.35rem">📅 Year</div>',
            unsafe_allow_html=True,
        )
        sel_year_label = st.selectbox(
            "Year", list(YEAR_OPTIONS.keys()),
            label_visibility="collapsed",
            key="filter_year",
        )

    with fc4:
        st.markdown(
            '<div class="crew-lbl" style="margin-bottom:.35rem">⏱️ Runtime</div>',
            unsafe_allow_html=True,
        )
        sel_runtime_label = st.selectbox(
            "Runtime", list(RUNTIME_OPTIONS.keys()),
            label_visibility="collapsed",
            key="filter_runtime",
        )

    st.markdown("</div>", unsafe_allow_html=True)

    # Resolve range tuples
    yr_lo,  yr_hi  = YEAR_OPTIONS[sel_year_label]
    rt_lo,  rt_hi  = RUNTIME_OPTIONS[sel_runtime_label]

    # ── Filter & sort ─────────────────────────────────────────────────────
    def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
        f = df.copy()

        # Year filter
        f = f[
            (f["release_date"].dt.year >= yr_lo) &
            (f["release_date"].dt.year <= yr_hi)
        ]

        # Runtime filter
        f = f[
            (f["runtime"] >= rt_lo) &
            (f["runtime"] <= rt_hi)
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
                        st.markdown(
                            movie_card_html(
                                mv["title"],
                                poster_url(mv.get("poster_path")),
                                rating=mv["averageRating"],
                                year=year,
                            ),
                            unsafe_allow_html=True,
                        )
            st.markdown("<div style='margin-bottom:.55rem'/>", unsafe_allow_html=True)
