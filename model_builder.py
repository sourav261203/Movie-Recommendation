import pandas as pd
import ast
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load Data
df_movies = pd.read_csv("Dataset/movie_dataset.csv")
df_movies.dropna(inplace=True)

# 2. Helper Functions
def tolist(text):
    if isinstance(text, str):
        return [i.strip() for i in text.split(",")]
    return []

def collapse(L):
    return [i.replace(" ", "") for i in L]

# 3. Create df_popular for Section 3 (Filtering)
df_popular = df_movies[['id', 'title', 'averageRating', 'numVotes', 'release_date', 'runtime', 'adult',
                        'popularity', 'poster_path', 'genres', 'production_countries', 'overview']].copy()

df_popular['genres'] = df_popular['genres'].apply(tolist)
df_popular['production_countries'] = df_popular['production_countries'].apply(tolist)
df_popular["release_date"] = pd.to_datetime(df_popular["release_date"], errors="coerce")

# 4. Create df_recommend for Sections 1 & 2 (Recommendations & Description)
df_recommend = df_movies[['id', 'title', 'overview', 'poster_path', 'genres', 'keywords', 'directors', 'writers', 'cast', 'release_date', 'runtime', 'averageRating']].copy()

df_recommend["genres"] = df_recommend["genres"].apply(tolist).apply(collapse)
df_recommend["keywords"] = df_recommend["keywords"].apply(tolist).apply(collapse)
df_recommend["directors"] = df_recommend["directors"].apply(tolist).apply(collapse)
df_recommend["writers"] = df_recommend["writers"].apply(tolist).apply(collapse)
df_recommend["cast"] = df_recommend["cast"].apply(tolist).apply(collapse)

# Create Tags
df_recommend['tags'] = (
    df_recommend['overview'].apply(lambda x: str(x).split()) + 
    df_recommend['genres'] + 
    df_recommend['keywords'] + 
    df_recommend['cast'].apply(lambda x: x[:5]) + 
    df_recommend['directors'].apply(lambda x: x[:1]) + 
    df_recommend['writers'].apply(lambda x: x[:1])
)
df_recommend['tags'] = df_recommend['tags'].apply(lambda x: " ".join(x).lower())

# 5. Vectorization and Similarity
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vector = tfidf.fit_transform(df_recommend['tags']).toarray()
similarity = cosine_similarity(vector)

# 6. Save the models
pickle.dump(df_popular.to_dict(), open('df_popular.pkl', 'wb'))
pickle.dump(df_recommend.to_dict(), open('df_recommend.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
print("Pipeline Execution Complete. Files saved successfully!")