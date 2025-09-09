from src.imports import *
from src.data_prep import get_genre_list, create_genre_dict

def boxplots_numeric(df, numeric_cols, save_dir='reports/figures/'):

    os.makedirs(save_dir, exist_ok=True)
    
    for col in numeric_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.xlabel(col)
        plt.tight_layout()
        
        plt.show()


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


def genre_analysis(df: pd.DataFrame, top_n: int = 10):
    genre_cols = [col for col in df.columns if col in get_genre_list()]
    if not genre_cols:
        print("Nenhum gênero encontrado no DataFrame.")
        return

    genre_counts = df[genre_cols].sum().sort_values(ascending=False).head(top_n)
    plt.figure(figsize=(12,6))
    sns.barplot(x=genre_counts.index, y=genre_counts.values, palette='pastel')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Top {top_n} Genres by Number of Movies')
    plt.xlabel('Genre')
    plt.ylabel('Number of Movies')
    plt.tight_layout()
    plt.show()


    if 'Gross_USD' in df.columns:
        genre_gross = {}
        for genre in genre_cols:
            temp = df[df[genre]==1]['Gross_USD'].dropna()
            temp = pd.to_numeric(temp, errors='coerce')
            genre_gross[genre] = temp.sum()
        genre_gross = pd.Series(genre_gross).sort_values(ascending=False).head(top_n)

        print(f"Total Gross Revenue by Top {top_n} Genres:")
        print(genre_gross)

        plt.figure(figsize=(12,6))
        sns.barplot(x=genre_gross.index, y=genre_gross.values, palette='pastel')
        plt.xticks(rotation=45, ha='right')
        plt.title(f'Top {top_n} Genres by Total Gross Revenue')
        plt.ylabel('Total Gross Revenue (USD)')
        plt.tight_layout()
        plt.show()


def scatter_votes_rating_by_genre(df):
    genre_cols = [c for c in get_genre_list() if c in df.columns]

    genre_counts = {g: df[g].sum() for g in genre_cols}
    top_genres = sorted(genre_counts, key=genre_counts.get, reverse=True)[:10]

    cols = 3
    rows = int(np.ceil(len(top_genres) / cols))
    plt.figure(figsize=(cols*5, rows*4))

    for i, g in enumerate(top_genres):
        mask = df[g] == 1
        x = df.loc[mask, 'No_of_Votes'].dropna()
        y = df.loc[mask, 'IMDB_Rating'].dropna()
        meta = df.loc[mask, 'Meta_score'].dropna()

        common_idx = x.index.intersection(y.index).intersection(meta.index)
        x = x.loc[common_idx]
        y = y.loc[common_idx]
        meta = meta.loc[common_idx]

        ax = plt.subplot(rows, cols, i+1)

        scatter = ax.scatter(x, y, s=15, alpha=0.5, c=meta, cmap='viridis')

        # LOWESS
        if len(x) > 0 and len(y) > 0:
            smoothed = lowess(y, x, frac=0.3)
            ax.plot(smoothed[:,0], smoothed[:,1], color='red', linewidth=2, label='LOWESS')

        ax.set_xscale('log')
        ax.set_title(g)
        ax.set_xlabel('Number of Votes')
        ax.set_ylabel('IMDB Rating')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.colorbar(scatter, label='Meta Score')
    plt.show()


def rating_vs_gross(df: pd.DataFrame, bins: int = 5):
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ensure numeric
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

    # Drop rows with missing values
    df_clean = df.dropna(subset=['Gross_USD', 'IMDB_Rating', 'Meta_score'])

    genre_cols = [col for col in df.columns if col in get_genre_list()]
    if genre_cols:
        df_clean['Main_Genre'] = df_clean[genre_cols].idxmax(axis=1)
    else:
        df_clean['Main_Genre'] = 'Unknown'

    # Correlation
    corr_imdb = df_clean['Gross_USD'].corr(df_clean['IMDB_Rating'])
    corr_meta = df_clean['Gross_USD'].corr(df_clean['Meta_score'])
    print(f"Correlation Gross_USD vs IMDB_Rating: {corr_imdb:.2f}")
    print(f"Correlation Gross_USD vs Meta_score: {corr_meta:.2f}")

    plt.figure(figsize=(12,5))

    # IMDB Rating
    plt.subplot(1,2,1)
    sns.scatterplot(
        x='IMDB_Rating', y='Gross_USD',
        data=df_clean,
        hue='Main_Genre',
        palette='tab20',
        alpha=0.6,
        s=50
    )
    plt.title('Gross Revenue vs IMDB Rating')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')
    plt.legend(title='Gênero', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Meta Score
    plt.subplot(1,2,2)
    sns.scatterplot(
        x='Meta_score', y='Gross_USD',
        data=df_clean,
        hue='Main_Genre',
        palette='tab20',
        alpha=0.6,
        s=50
    )
    plt.title('Gross Revenue vs Meta Score')
    plt.ylabel('Gross Revenue (USD)')
    plt.yscale('log')
    plt.legend([],[], frameon=False)  # remove legenda duplicada

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


def boxplots_by_year_quartile(df: pd.DataFrame, year_col="Released_Year", vote_col="No_of_Votes", gross_col="Gross_USD"):
    df = df.copy()
    df["year_quartile"] = pd.qcut(df[year_col], 4, labels=["Q1 - Oldest", "Q2", "Q3", "Q4 - Newest"])
    
    agg_stats = df.groupby("year_quartile")[[vote_col, gross_col]].mean().reset_index()
    print(agg_stats)
    
    fig, axes = plt.subplots(1, 2, figsize=(12,5))
    
    # Boxplot Votes
    df.boxplot(column=vote_col, by="year_quartile", ax=axes[0])
    axes[0].set_title(f"{vote_col} by Release Year Quartiles")
    axes[0].set_ylabel(vote_col)
    
    # Boxplot Gross
    df.boxplot(column=gross_col, by="year_quartile", ax=axes[1])
    axes[1].set_title(f"{gross_col} by Release Year Quartiles")
    axes[1].set_ylabel(gross_col)
    
    plt.suptitle("")  
    plt.tight_layout()
    plt.show()


def stars_appearances_vs_gross(df: pd.DataFrame, stars=['Star1','Star2','Star3','Star4'], top_n=10):
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
    # Ensure numeric ratings
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    
    # --- Top directors by number of movies ---
    fig, ax = plt.subplots(figsize=(20,5))
    top_directors_count = df['Director'].value_counts().head(top_n)
    sns.barplot(x=top_directors_count.index, y=top_directors_count.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Number of Movies", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Number of Movies", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    
    # --- Top directors by average IMDb Rating ---
    avg_imdb = df.groupby('Director')['IMDB_Rating'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(20,5))
    sns.barplot(x=avg_imdb.index, y=avg_imdb.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Average IMDB Rating", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Average IMDB Rating", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()
    
    # --- Top directors by average Meta_score ---
    avg_meta = df.groupby('Director')['Meta_score'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(20,5))
    sns.barplot(x=avg_meta.index, y=avg_meta.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Average Meta_score", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Average Meta_score", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def directors_financial_analysis(df, top_n=10):
    # Ensure numeric columns
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')

    # --- Top directors by average gross ---
    avg_gross = df.groupby('Director')['Gross_USD'].mean().sort_values(ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(20,5))
    sns.barplot(x=avg_gross.index, y=avg_gross.values, palette='pastel', ax=ax)
    ax.set_title(f"Top {top_n} Directors by Average Gross Revenue", weight="bold")
    ax.set_xlabel("Directors", weight="bold")
    ax.set_ylabel("Average Gross (USD)", weight="bold")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

    # --- Scatterplot IMDb × Meta_score with gross size ---
    directors_stats = df.groupby('Director').agg({
        'IMDB_Rating': 'mean',
        'Meta_score': 'mean',
        'Gross_USD': 'mean'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10,7))
    sns.scatterplot(
        data=directors_stats,
        x='IMDB_Rating',
        y='Meta_score',
        size='Gross_USD',
        sizes=(50, 800),
        alpha=0.6,
        legend=False
    )
    ax.set_title("Directors: IMDb vs Meta_score (Bubble = Avg Gross)", weight="bold")
    ax.set_xlabel("Average IMDb Rating", weight="bold")
    ax.set_ylabel("Average Meta_score", weight="bold")
    plt.tight_layout()
    plt.show()
   

def top_movies_analysis(df: pd.DataFrame, top_n=10):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    # Ensure numeric columns
    df['Gross_USD'] = pd.to_numeric(df['Gross_USD'], errors='coerce')
    df['IMDB_Rating'] = pd.to_numeric(df['IMDB_Rating'], errors='coerce')
    df['Meta_score'] = pd.to_numeric(df['Meta_score'], errors='coerce')

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


