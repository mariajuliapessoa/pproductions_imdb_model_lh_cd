import sys
from pathlib import Path
import pandas as pd
import numpy as np
import re
from unidecode import unidecode

project_root = Path("..").resolve()
sys.path.append(str(project_root))

# Load dataset from CSV
def load_data(file_path="data/raw/desafio_indicium_imdb.csv"):
    project_root = Path(__file__).parent.parent
    file_path = project_root / file_path
    if not file_path.exists():
        raise FileNotFoundError(f"File not found at {file_path.resolve()}")
    return pd.read_csv(file_path)

# Return percentage of missing values per column
def missing_values_summary(df):
    return df.isna().mean().sort_values(ascending=False) * 100

# Rename specific columns to standardized names
def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    columns_map = {
        "Series_Title": "Movie_title",
        "Unnamed: 0": "Index"
    }
    return df.rename(columns=columns_map)


# Check for duplicates based on a subset of columns
def check_duplicates(df, subset):
    if all(k in df.columns for k in subset):
        duplicates = df.duplicated(subset=subset, keep=False)
        return df.loc[duplicates].sort_values(subset) if duplicates.sum() > 0 else None
    else:
        raise ValueError(f"Missing columns for duplicate check: {subset}")

# Safely convert a value to pandas Int64 type 
def to_int64_safe(x):
    try:
        return pd.Int64Dtype().type(int(x))
    except Exception:
        try:
            return pd.Int64Dtype().type(int(float(x)))
        except Exception:
            return pd.NA

# Convert gross revenue string to float
def parse_gross_to_float(val):
    if pd.isna(val):
        return np.nan
    s = re.sub(r'[^0-9]', '', str(val))
    return float(s) if s else np.nan

# Clean and standardize string columns
def clean_text_columns(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = ['Series_Title', 'Certificate', 'Overview', 'Director',
                'Star1', 'Star2', 'Star3', 'Star4']
    
    df_clean = df.copy()
    
    for col in cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype(str).str.strip()
    
    return df_clean

# Standardize movie certificate ratings
def normalize_certificate(cert):
    if pd.isna(cert) or str(cert).strip() == '':
        return 'UNRATED'

    cert_clean = unidecode(str(cert)).strip().upper().replace(' ', '').replace('.', '')

    cert_dict = {
        'U': 'G', 'A': 'PG', 'UA': 'PG', 'U/A': 'PG', 'APPROVED': 'PG',
        'PG': 'PG', 'PG-13': 'PG-13', 'R': 'R', 'G': 'G', 'PASSED': 'PG',
        'TV-14': 'PG-13', 'TV-MA': 'R', 'TV-PG': 'PG', 'UNRATED': 'UNRATED',
        'GP': 'PG'
    }

    return cert_dict.get(cert_clean, 'UNRATED')

# Split genre string into a list
def split_genres(x):
    if pd.isna(x):
        return []
    return [g.strip() for g in str(x).split(',')]

# Return top n most frequent values in a series
def top_n(series, n=10):
    return series.value_counts().head(n)

# plots.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_gross_and_rating(df, top_n=10):
    """
    Plots average gross and average rating for the top N most frequent genres.

    Parameters:
    df : pandas.DataFrame
        DataFrame containing columns 'Genre', 'Gross_USD', 'IMDB_Rating', and 'Meta_score'.
    top_n : int
        Number of most frequent genres to consider (default=10).
    """

    # Select the top N most frequent genres
    top_genres = df['Genre'].value_counts().head(top_n).index
    df_top = df[df['Genre'].isin(top_genres)]

    # Calculate total and average gross per genre
    gross_by_genre = df_top.groupby('Genre')['Gross_USD'].agg(['sum', 'mean']).sort_values('sum', ascending=False)
    print("Gross per Genre (sum & mean):")
    print(gross_by_genre)

    # Calculate average IMDB rating and Meta Score per genre
    rating_by_genre = df_top.groupby('Genre')[['IMDB_Rating', 'Meta_score']].mean().sort_values('IMDB_Rating', ascending=False)
    print("\nAverage Ratings per Genre (IMDB & Meta Score):")
    print(rating_by_genre)

    # --- Plot average gross per genre ---
    plt.figure(figsize=(10,5))
    sns.barplot(x='mean', y=gross_by_genre.index, data=gross_by_genre.reset_index(), palette='viridis')
    plt.title(f'Average Gross per Genre (Top {top_n})')
    plt.xlabel('Average Gross (USD)')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()

    # --- Plot average IMDB rating per genre ---
    plt.figure(figsize=(10,5))
    rating_by_genre_sorted = rating_by_genre.sort_values('IMDB_Rating', ascending=False)
    sns.barplot(x='IMDB_Rating', y=rating_by_genre_sorted.index, data=rating_by_genre_sorted.reset_index(), palette='coolwarm')
    plt.title(f'Average IMDB Rating per Genre (Top {top_n})')
    plt.xlabel('IMDB Rating')
    plt.ylabel('Genre')
    plt.tight_layout()
    plt.show()

