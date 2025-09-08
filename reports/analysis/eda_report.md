# IMDb Top 1000 Movies – Exploratory Data Analysis (EDA)

## Cover / Introduction
This report presents a detailed Exploratory Data Analysis (EDA) of the IMDb Top 1000 movies dataset. The goal of this analysis is to provide a comprehensive understanding of the dataset, including data quality, feature distributions, missing values, and initial insights that can guide further predictive modeling and recommendation system development.

## Context of the Challenge
The challenge involves analyzing the IMDb Top 1000 movies dataset, augmented with additional external data sources. Key objectives include:

- Understanding the structure and quality of the data.
- Identifying missing values and data inconsistencies.
- Performing feature engineering to extract meaningful insights.
- Conducting exploratory visualizations to guide future modeling efforts.

## Dataset Description
The main dataset includes information on 1000 top-rated movies according to IMDb, along with external financial and categorical data. The key features include:

| Feature Name        | Description |
|-------------------|-------------|
| Index              | Unique identifier for each movie |
| Movie_title        | Title of the movie |
| Released_Year      | Year of release |
| Certificate        | MPAA rating (e.g., PG, R, UNRATED) |
| Genre              | Movie genre(s) |
| IMDb_Rating        | IMDb user rating (scale 0–10) |
| Meta_score         | MetaCritic score |
| Runtime_Min        | Duration in minutes |
| Gross_USD          | Box office revenue in USD |
| Director           | Director name |
| Star1–Star4        | Main cast members |
| Overview           | Movie synopsis |
| No_of_Votes        | Number of IMDb votes |

### Data Sources
- Original IMDb dataset (`data/raw/desafio_indicium_imdb.csv`)  

## 1. Data Loading and Cleaning
The dataset was loaded using a custom `load_data()` function. Initial data checks included:

- Verifying column data types.
- Inspecting the number of missing values per column using `missing_values_summary()`.
- Renaming columns to standardized names:
  - `Series_Title` → `Movie_title`
  - `Unnamed: 0` → `Index`
- Cleaning text columns (`Movie_title`, `Certificate`, `Overview`, `Director`, `Star1–Star4`) by stripping leading/trailing whitespace.
- Standardizing categorical fields (e.g., `Certificate` normalized via `normalize_certificate()`).
- Numeric conversions:
  - `Released_Year` and `Runtime_Min` to integers.
  - Gross revenue strings parsed to floats.

## 2. Handling Missing Data
Missing values were analyzed for all columns. Key observations include:

- `Meta_score` has approximately 15% missing values.
- `Gross_USD` is missing mainly for older movies.
- Certain categorical fields (e.g., `Certificate`) contain blank or missing entries.
- Numeric missing values were retained as `NaN` for potential imputation; categorical columns were normalized or filled with default values such as `"UNRATED"`.

## 3. Feature Engineering
Several new features were derived:

### Text Features
- `Overview_len`: Number of characters in the movie overview.
- `Overview_words`: Number of words in the movie overview.

### Genre Handling
- `Genre` column split into a list of genres (`split_genres`).
- Multi-hot encoding applied using `MultiLabelBinarizer` to create individual binary columns per genre.

### Numeric Conversions
- `Released_Year`, `Runtime_Min`, and `Gross_USD` converted to numeric types.

### Outlier Handling
- Boxplots were used to visually inspect outliers in numeric columns.
![IMDB_Rating Outliers](<../figures/Boxplot of Gross_USD (outliers).png>)
![Gross_USD Outliers](<../figures/Boxplot of Gross_USD (outliers).png>)
![Meta_Score Outliers](<../figures/Boxplot of Meta_Score (outliers).png>)
![No_of_Votes Outliers](<../figures/Boxplot of No_of_Votes (outliers).png>)
![Runtime_Min](<../figures/Boxplot of Runtime_Min (outliers).png>)


## 4. Exploratory Visualization

![Exploring the Distribution of Movie Certificates](<../figures/Distribution of Movies Certificates.png>)

The “Distribution of Movie Certificates” chart reveals an uneven distribution among movie ratings: PG is the most frequent, followed by G, while PG-13, R, and Unrated are less common. This indicates that the dataset is predominantly composed of family-friendly and general audience films, which could bias analyses of box office performance, audience reach, and critical reception. Plausible hypotheses include: PG-rated films tend to attract larger audiences, reflected in higher vote counts and total gross; R-rated films may have higher IMDb or Meta scores but a more limited audience; and certificate level may influence the relationship between genre and revenue. Recommended next steps include cross-tabulations between Certificate and Genre or Gross, descriptive statistics by Certificate, and additional visualizations such as boxplots of Gross or average votes per Certificate. Certificates should also be encoded for predictive modeling to assess their impact on financial or popularity metrics. Attention should be given to temporal biases and the meaning of “Unrated” entries to ensure correct interpretation.

### 4.1 Histograms

### 4.2 Barplots

![Total Gross Revenue by Top Genres](<../figures/Total Gross Revenue by Top Genres.png>)

The analysis of movie genres reveals that while Drama is the most common genre, it is not the one generating the highest revenue. Adventure movies lead in total gross, closely followed by Drama, indicating that even though Drama is more frequent, it still achieves substantial earnings. The top five genres in terms of revenue are, in descending order: Adventure, Drama, Action, Comedy, and Sci-Fi. This suggests that while popularity in terms of number of movies does not always align perfectly with financial success, there is a strong overlap, with frequently produced genres like Drama still performing well at the box office.


### 4.2 Boxplots

Drama has been identified as the most predominant genre in the dataset. This analysis will further investigate whether Drama also leads in terms of box office revenue and average ratings (IMDB and Meta Score) compared to other genres, providing insights into its overall commercial and critical performance."


### 4.3 Correlation Analysis

The correlation analysis reveals that the number of votes shows the strongest association with box office gross (r = 0.59), suggesting that popularity is a stronger driver of revenue than ratings or critic scores. Interestingly, critic evaluations (Meta_score) show almost no relationship with commercial success, while audience ratings (IMDB_Rating) exhibit only a weak correlation with gross. This highlights a potential gap between perceived quality and financial performance, emphasizing that popularity metrics may be more relevant for predicting box office outcomes.

Seção “Hypotheses” ou “Future Work”:

Test whether votes mediate the relationship between ratings and gross.

Explore temporal effects (e.g., older movies having accumulated more votes).

Segment correlations by genre to see if some genres depend more on critical reception.

Star Power Analysis: Appearances vs Total Gross

The analysis of the top actors in each primary cast position (Star1, Star2, Star3, Star4) reveals key insights about the relationship between frequency of appearances and box office performance.

For Star1, the first panel shows the top 10 actors by appearances, highlighting who most frequently occupies the leading role. The adjacent panel presents the same actors ranked by total gross revenue, showing which Star1 actors contributed most to overall revenue. Notably, some actors appear frequently but generate only moderate revenue, suggesting that high screen presence alone does not guarantee strong box office returns. Conversely, certain actors achieve high revenue when appearing as Star1, indicating strong market value even with fewer appearances.

A similar pattern emerges for Star2, Star3, and Star4. While secondary and tertiary roles generally have less impact than Star1, comparing appearances and total gross reveals that some actors consistently drive significant revenue across different positions. This demonstrates that star power is not solely dependent on frequency of appearances but also on the impact of the films they participate in.

Practical insights for casting and business decisions include:

Prioritize actors who generate high total gross when appearing as Star1 or Star2, particularly if the project is in the same genre or franchise.

Consider the position of the actor within the cast, as Star1 roles tend to influence revenue more strongly than Star3 or Star4.

Complement this analysis with additional features, such as genre, release year, and budget, to understand the contexts in which star power translates into higher revenue.

## 5. Key Insights
- IMDb ratings are narrowly distributed between 7.0 and 9.2.  
- Most movies belong to Drama, Comedy, and Adventure genres.  
- Older movies tend to have missing financial data (`Gross_USD`).  
- Multi-label genre encoding allows machine learning models to capture overlapping genres.  
- Outlier inspection suggests applying log transformations for skewed numeric data.

## 6. Conclusion
This EDA provides a comprehensive overview of the IMDb Top 1000 movies dataset. Key preprocessing steps, missing data handling, feature engineering, and initial visual insights have been documented. These results lay the groundwork for:

- Further data modeling.
- Predictive analysis.
- Recommendation system development.



### Recommendation Based on Ratings

To recommend a movie to someone I don’t know, I would prioritize **highly rated films**, as they are likely to appeal to a broader audience and provide quality entertainment.

* **Approach**: I looked at the dataset and identified the films with the highest `IMDB_Rating` and `Meta_score`.
* **Observation**: The top-rated movie(s) consistently score very high in both audience and critic evaluations, indicating universal appeal.
* **Recommendation**: The movie with the highest `IMDB_Rating` (and also strong `Meta_score`) would be my pick for someone unfamiliar with the dataset or their preferences.

**Example statement**:
*"Based on our analysis, the most highly rated film in the dataset is \[MOVIE\_TITLE], which has an IMDB rating of \[IMDB\_RATING] and a Meta\_score of \[META\_SCORE]. This makes it the safest recommendation for general audiences, combining critical acclaim with audience satisfaction."*





