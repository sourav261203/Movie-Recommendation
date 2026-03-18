# 🎬 CineScope — Movie Recommender

A production-ready movie recommendation system with a cinematic Streamlit UI.

## Project Structure

```
movie_recommender/
├── app.py              # Streamlit application (4 pages)
├── model_builder.py    # Pipeline: CSV → clean → TF-IDF → cosine similarity → .pkl
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the app
```bash
streamlit run app.py
```

### 3. Build the model (in-app)
1. Go to **⚙️ Build Model** tab
2. Upload your `movie_dataset.csv`
3. Adjust parameters (TF-IDF features, output directory)
4. Click **Build & Save Models**
5. The app will generate `df_popular.pkl`, `df_recommend.pkl`, `similarity.pkl`

### Alternative: CLI build
```bash
python model_builder.py --dataset Dataset/movie_dataset.csv --output .
python model_builder.py --dataset my_data.csv --output ./models --max-features 8000
```

## Pages

| Page | Description |
|---|---|
| ⚙️ Build Model | Upload CSV, run the full pipeline, save .pkl files |
| 🎯 Recommend Me | Content-based top-10 similar movie recommendations |
| 🎞️ Movie Details | Full metadata, crew info, genre badges, overview |
| 🔥 Top Movies | Filter by genre, country, year, runtime, votes |

## Dataset Requirements

Your CSV must contain these columns:

```
id, title, overview, poster_path, genres, keywords,
directors, writers, cast, release_date, runtime,
averageRating, numVotes, popularity, adult, production_countries
```

**Note:** Only rows with **no null values** are used (via `dropna()`).

## Model Details

- **Vectorisation**: TF-IDF on composite tags (overview + genres + keywords + cast + director + writer)
- **Similarity**: Cosine similarity matrix (movies × movies)
- **Filtering pipeline**: Popularity + vote threshold + genre/country/year/runtime filters
- **Posters**: Fetched live from TMDB (`https://image.tmdb.org/t/p/w500/`)
