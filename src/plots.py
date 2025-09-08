from src.imports import *

# plots.py
import os

def boxplots_numeric(df, numeric_cols, save_dir='reports/figures/'):

    os.makedirs(save_dir, exist_ok=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.tight_layout()
        
        plt.show()

# plots.py
import os
import matplotlib.pyplot as plt
import seaborn as sns

def barplot_certificate_distribution(df, cert_order=None, save_dir='reports/figures/'):
    if cert_order is None:
        cert_order = ['G','PG','PG-13','R','UNRATED']
    
    # Reindex counts to ensure consistent order
    cert_counts = df['Certificate'].value_counts().reindex(cert_order)

    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(8,5))
    sns.barplot(x=cert_counts.index, y=cert_counts.values, palette='pastel')
    plt.title('Distribution of Movie Certificates')
    plt.xlabel('Certificate')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    
    plt.show()


# Set general plotting style
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def plot_hist(df: pd.DataFrame, col: str, bins: int = 30, log_scale: bool = False):
    if col not in df.columns:
        print(f"Column {col} not found in DataFrame.")
        return
    
    data = df[col].dropna()
    plt.figure(figsize=(10,5))
    sns.histplot(data, bins=bins, kde=True)
    
    if log_scale:
        plt.yscale('log')
    
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()



def plot_corr_heatmap(df: pd.DataFrame, numeric_cols: list = None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['float64', 'Int64']).columns.tolist()
    
    if not numeric_cols:
        print("No numeric columns found to plot.")
        return
    
    plt.figure(figsize=(15,10))
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

# src/plots.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def genre_analysis(df: pd.DataFrame):
    """
    Generate visual analysis of movie genres using multi-hot encoded genre columns.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing multi-hot encoded genre columns and Gross_USD.

    Outputs:
    --------
    1. Barplot of number of movies per genre
    2. Pie chart of top 10 genres
    3. Boxplot of Gross revenue per top 10 genre
    """

    # Identify multi-hot genre columns
    genre_cols = [col for col in df.columns if col in [
        'Drama', 'Comedy', 'Crime', 'Adventure', 'Action', 'Thriller', 'Romance', 
        'Biography', 'Mystery', 'Animation', 'Sci-Fi', 'Fantasy', 'Family', 'History',
        'War', 'Music', 'Horror', 'Western', 'Film-Noir', 'Sport'
    ]]

    if not genre_cols:
        print("No multi-hot genre columns found in DataFrame.")
        return

    # Barplot: total movies per genre
    genre_counts = df[genre_cols].sum().sort_values(ascending=False)
    plt.figure(figsize=(12,6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.title('Distribution of Movie Genres (Multi-hot)')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    plt.show()

    # Pie chart: top 10 genres
    top_genres = genre_counts.head(10)
    plt.figure(figsize=(8,8))
    plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('pastel'))
    plt.title('Top 10 Genres Share (Multi-hot)')
    plt.show()

    # Boxplot: Gross revenue by top 10 genres
    top_genres_list = top_genres.index.tolist()
    df_box = []

    for genre in top_genres_list:
        if 'Gross_USD' not in df.columns:
            continue
        temp = df[df[genre]==1][['Gross_USD']].copy()
        temp = temp.dropna(subset=['Gross_USD'])           # remove NaNs
        temp['Genre'] = genre
        df_box.append(temp)

    if not df_box:
        print("No data available for Gross_USD boxplot.")
        return

    # Concat all genres together
    df_box = pd.concat(df_box, ignore_index=True)

    # Ensure numeric type for Gross_USD
    df_box['Gross_USD'] = pd.to_numeric(df_box['Gross_USD'], errors='coerce')
    df_box = df_box.dropna(subset=['Gross_USD'])

    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_box, x='Genre', y='Gross_USD', palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.title('Gross Revenue by Genre (Top 10, Multi-hot)')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')  # optional: better visualization of outliers
    plt.tight_layout()
    plt.show()

def genre_gross_analysis(df: pd.DataFrame, top_n: int = 10):

    # Identify multi-hot genre columns
    genre_cols = [col for col in df.columns if col in [
        'Drama', 'Comedy', 'Crime', 'Adventure', 'Action', 'Thriller', 'Romance', 
        'Biography', 'Mystery', 'Animation', 'Sci-Fi', 'Fantasy', 'Family', 'History',
        'War', 'Music', 'Horror', 'Western', 'Film-Noir', 'Sport'
    ]]

    if not genre_cols:
        print("No multi-hot genre columns found in DataFrame.")
        return

    # Sum Gross_USD for each genre
    genre_gross = {}
    for genre in genre_cols:
        temp = df[df[genre]==1]['Gross_USD'].dropna()
        temp = pd.to_numeric(temp, errors='coerce')
        genre_gross[genre] = temp.sum()

    # Convert to Series and get top N genres by total gross
    genre_gross = pd.Series(genre_gross).sort_values(ascending=False).head(top_n)
    print("Total Gross Revenue by Top Genres:")
    print(genre_gross)

    # Barplot: total gross per genre
    plt.figure(figsize=(12,6))
    sns.barplot(x=genre_gross.index, y=genre_gross.values, palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Total Gross Revenue by Top {top_n} Genres (Multi-hot)')
    plt.ylabel('Total Gross Revenue (USD)')
    plt.tight_layout()
    plt.show()

    # Boxplot: distribution of gross per genre
    df_box = []
    for genre in genre_gross.index:
        temp = df[df[genre]==1][['Gross_USD']].copy()
        temp = temp.dropna(subset=['Gross_USD'])
        temp['Genre'] = genre
        df_box.append(temp)

    df_box = pd.concat(df_box, ignore_index=True)

    plt.figure(figsize=(12,6))
    sns.boxplot(data=df_box, x='Genre', y='Gross_USD', palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Gross Revenue Distribution by Top {top_n} Genres (Multi-hot)')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')  # optional: better visualization of outliers
    plt.tight_layout()
    plt.show()

def rating_vs_gross(df: pd.DataFrame, bins: int = 5):
    """
    Analyze whether higher-rated movies generate higher gross revenue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Gross_USD', 'IMDB_Rating', and 'Meta_score'.
    bins : int
        Number of bins for average gross by rating.

    Outputs:
    --------
    1. Scatter plots of Gross_USD vs IMDB_Rating and Meta_score
    2. Print correlation coefficients
    3. Bar plots of average gross by rating bins
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np

    # Ensure numeric
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

    # Drop rows with missing values
    df_clean = df.dropna(subset=['Gross_USD', 'IMDB_Rating', 'Meta_score'])

    # Correlation
    corr_imdb = df_clean['Gross_USD'].corr(df_clean['IMDB_Rating'])
    corr_meta = df_clean['Gross_USD'].corr(df_clean['Meta_score'])
    print(f"Correlation Gross_USD vs IMDB_Rating: {corr_imdb:.2f}")
    print(f"Correlation Gross_USD vs Meta_score: {corr_meta:.2f}")

    # Scatter plots
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.scatterplot(x='IMDB_Rating', y='Gross_USD', data=df_clean)
    plt.title('Gross Revenue vs IMDB Rating')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')  # log scale for better visualization
    plt.subplot(1,2,2)
    sns.scatterplot(x='Meta_score', y='Gross_USD', data=df_clean)
    plt.title('Gross Revenue vs Meta Score')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')
    plt.tight_layout()
    plt.show()

    # Average gross by rating bins
    for col in ['IMDB_Rating', 'Meta_score']:
        df_clean[f'{col}_bin'] = pd.qcut(df_clean[col], bins, duplicates='drop')
        avg_gross = df_clean.groupby(f'{col}_bin')['Gross_USD'].mean()
        plt.figure(figsize=(8,4))
        sns.barplot(x=avg_gross.index.astype(str), y=avg_gross.values, palette='pastel')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Average Gross by {col} Bins')
        plt.ylabel('Average Gross (USD)')
        plt.tight_layout()
        plt.show()


def star_gross_analysis(df: pd.DataFrame, top_n: int = 10):
    """
    Analyze whether top-billed actors (Star1) influence gross revenue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Star1' and 'Gross_USD'.
    top_n : int
        Number of top actors to display (default=10).

    Outputs:
    --------
    1. Prints top actors by total and average gross revenue
    2. Bar plot of top actors by total gross
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure numeric Gross_USD
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')

    # Drop missing values
    df_clean = df.dropna(subset=['Star1', 'Gross_USD'])

    # Group by Star1
    star_stats = df_clean.groupby('Star1')['Gross_USD'].agg(['sum', 'mean', 'count']).sort_values('sum', ascending=False)

    print("Top actors by total gross revenue:")
    print(star_stats.head(top_n))

    # Bar plot of total gross for top N actors
    top_stars = star_stats.head(top_n).reset_index()
    plt.figure(figsize=(12,6))
    sns.barplot(x='Star1', y='sum', data=top_stars, palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Total Gross Revenue (USD)')
    plt.title(f'Top {top_n} Actors by Total Gross Revenue')
    plt.tight_layout()
    plt.show()


def stars_appearances_vs_gross(df: pd.DataFrame, stars=['Star1','Star2','Star3','Star4'], top_n=10):
    """
    Compare the most frequent actors (appearances) with total gross revenue.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Star1', 'Star2', 'Star3', 'Star4' and 'Gross_USD'.
    stars : list
        List of columns representing main stars (default: ['Star1','Star2','Star3','Star4']).
    top_n : int
        Number of top actors to display for each star position.

    Outputs:
    --------
    1. Barplots for top actors by appearances
    2. Barplots for top actors by total gross revenue
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Ensure numeric Gross_USD
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')
    df_clean = df.dropna(subset=stars + ['Gross_USD'])

    fig, axs = plt.subplots(len(stars), 2, figsize=(20, 5*len(stars)))
    plt.subplots_adjust(hspace=0.5)

    for i, star_col in enumerate(stars):
        # Top appearances
        top_appearances = df_clean[star_col].value_counts().head(top_n)
        sns.barplot(x=top_appearances.index, y=top_appearances.values, ax=axs[i,0], palette='pastel')
        axs[i,0].set_title(f'Top {top_n} {star_col} by Appearances')
        axs[i,0].set_ylabel('Appearances')
        axs[i,0].tick_params(axis='x', rotation=45)

        # Top total gross
        gross_by_star = df_clean.groupby(star_col)['Gross_USD'].sum().sort_values(ascending=False).head(top_n)
        sns.barplot(x=gross_by_star.index, y=gross_by_star.values, ax=axs[i,1], palette='pastel')
        axs[i,1].set_title(f'Top {top_n} {star_col} by Total Gross')
        axs[i,1].set_ylabel('Total Gross (USD)')
        axs[i,1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def directors_analysis(df: pd.DataFrame, top_n=10):
    """
    Analyze directors in terms of number of movies and average ratings.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Director', 'IMDB_Rating', 'Meta_score'.
    top_n : int
        Number of top directors to display for each metric.

    Outputs:
    --------
    1. Barplot of top directors by number of movies
    2. Barplot of top directors by average IMDB rating
    3. Barplot of top directors by average Meta_score
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Ensure numeric ratings
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    
    # 1️⃣ Top directors by number of movies
    fig, ax = plt.subplots(figsize=(20,5))
    top_directors_count = df['Director'].value_counts().head(top_n)
    sns.barplot(x=top_directors_count.index, y=top_directors_count.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Number of Movies", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Number of Movies", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 2️⃣ Top directors by average IMDB rating
    avg_imdb = df.groupby('Director')['IMDB_Rating'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(20,5))
    sns.barplot(x=avg_imdb.index, y=avg_imdb.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Average IMDB Rating", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Average IMDB Rating", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    
    # 3️⃣ Top directors by average Meta_score
    avg_meta = df.groupby('Director')['Meta_score'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(20,5))
    sns.barplot(x=avg_meta.index, y=avg_meta.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Average Meta_score", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Average Meta_score", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()


def top_movies_analysis(df: pd.DataFrame, top_n=10):
    """
    Analyze top movies by gross revenue and ratings, and calculate correlations.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'Movie_title', 'Gross_USD', 'IMDB_Rating', 'Meta_score'.
    top_n : int
        Number of top movies to display.

    Outputs:
    --------
    1. Top movies by Gross_USD
    2. Top movies by IMDB_Rating
    3. Top movies by Meta_score
    4. Scatter plots of Gross vs IMDB and Gross vs Meta_score
    5. Correlation coefficients
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Ensure numeric columns
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

    # 1️⃣ Top movies by Gross_USD
    top_gross = df[['Movie_title','Gross_USD']].dropna().sort_values('Gross_USD', ascending=False).head(top_n)
    print(f"Top {top_n} movies by Gross_USD:")
    print(top_gross)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x='Gross_USD', y='Movie_title', data=top_gross, palette='pastel')
    plt.title(f"Top {top_n} Movies by Gross Revenue", weight='bold')
    plt.xlabel("Gross Revenue (USD)", weight='bold')
    plt.ylabel("Movie Title", weight='bold')
    plt.tight_layout()
    plt.show()

    # 2️⃣ Top movies by IMDB_Rating
    top_imdb = df[['Movie_title','IMDB_Rating']].dropna().sort_values('IMDB_Rating', ascending=False).head(top_n)
    print(f"\nTop {top_n} movies by IMDB Rating:")
    print(top_imdb)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x='IMDB_Rating', y='Movie_title', data=top_imdb, palette='pastel')
    plt.title(f"Top {top_n} Movies by IMDB Rating", weight='bold')
    plt.xlabel("IMDB Rating", weight='bold')
    plt.ylabel("Movie Title", weight='bold')
    plt.tight_layout()
    plt.show()

    # 3️⃣ Top movies by Meta_score
    top_meta = df[['Movie_title','Meta_score']].dropna().sort_values('Meta_score', ascending=False).head(top_n)
    print(f"\nTop {top_n} movies by Meta_score:")
    print(top_meta)
    
    plt.figure(figsize=(12,6))
    sns.barplot(x='Meta_score', y='Movie_title', data=top_meta, palette='pastel')
    plt.title(f"Top {top_n} Movies by Meta_score", weight='bold')
    plt.xlabel("Meta_score", weight='bold')
    plt.ylabel("Movie Title", weight='bold')
    plt.tight_layout()
    plt.show()

    # 4️⃣ Correlation between Gross and ratings
    corr_imdb = df[['Gross_USD','IMDB_Rating']].corr().iloc[0,1]
    corr_meta = df[['Gross_USD','Meta_score']].corr().iloc[0,1]

    print(f"\nCorrelation between Gross and IMDB Rating: {corr_imdb:.3f}")
    print(f"Correlation between Gross and Meta_score: {corr_meta:.3f}")

    # Scatter plots
    fig, axes = plt.subplots(1,2, figsize=(16,6))
    sns.scatterplot(x='IMDB_Rating', y='Gross_USD', data=df, ax=axes[0])
    axes[0].set_title("Gross vs IMDB Rating")
    axes[0].set_yscale('log')

    sns.scatterplot(x='Meta_score', y='Gross_USD', data=df, ax=axes[1])
    axes[1].set_title("Gross vs Meta_score")
    axes[1].set_yscale('log')

    plt.tight_layout()
    plt.show()

