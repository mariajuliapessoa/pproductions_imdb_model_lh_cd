# IMDB Rating Prediction – Technical Report

## 1. Problem Definition

The task is to predict the **IMDB rating** of a movie, which is a **continuous numeric variable**. Therefore, this is a **regression problem**.


## 2. Features and Transformations

* **Numeric features:**
  `Runtime_Min`, `Meta_score`, `Gross_USD`, `No_of_Votes`

  * Missing values were imputed using the median.
  * Standard scaling was applied to normalize ranges.

* **Categorical features:**
  `Certificate`

  * One-hot encoding was applied to convert categories into binary columns.

* **Text feature:**
  `Overview`

  * Transformed using **TF-IDF vectorization** (max 5000 features, English stopwords removed) to capture important keywords.

* **Genre:**

  * Already one-hot encoded into multiple binary columns (e.g., `Drama`, `Comedy`, `Action`, etc.).
  * Each movie’s genre is represented by 1 in the corresponding columns.


## 3. Model Selection

**Gradient Boosting Regressor** was chosen due to:

* **Pros:**

  * Handles mixed numeric and categorical features well.
  * Captures nonlinear relationships and interactions.
  * Robust to outliers.

* **Cons:**

  * Computationally heavier than linear models.
  * Harder to interpret.
  * Sensitive to hyperparameters.


## 4. Model Pipeline

```python
# Preprocessing
preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')),
                      ('scaler', StandardScaler())]), numeric_features),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                      ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features),
    ('bin', 'passthrough', binary_features),
    ('tfidf', TfidfVectorizer(max_features=5000, stop_words='english'), text_features)
])

# Final pipeline
model = Pipeline([
    ('preproc', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Train
model.fit(X_train, y_train)
```


## 5. Performance Metrics

* **RMSE:** 0.208

  * Provides the average prediction error in rating points.
* **R² Score:** 0.343

  * Indicates \~34% of variance explained by the model.

> RMSE was chosen because it penalizes large errors and is intuitive in the same scale as IMDB ratings. R² complements it by showing variance explained.


## 6. Genre Dictionary Function

```python
def create_genre_dict():
    genre_cols = ['Drama', 'Comedy', 'Crime', 'Adventure', 'Action', 'Thriller',
                  'Romance', 'Biography', 'Mystery', 'Animation', 'Sci-Fi', 'Fantasy',
                  'Family', 'History', 'War', 'Music', 'Horror', 'Western', 'Film-Noir', 'Sport']
    return {genre: 0 for genre in genre_cols}

genre_dict = create_genre_dict()
```


## 7. Prediction Function

```python
def predict_imdb_rating(movie_dict, model, genre_cols):
    df = pd.DataFrame(columns=['Runtime_Min','Meta_score','Gross_USD','No_of_Votes','Certificate','Overview']+genre_cols)
    
    # Fill features
    df.at[0,'Runtime_Min'] = movie_dict.get('Runtime_Min')
    df.at[0,'Meta_score'] = movie_dict.get('Meta_score')
    df.at[0,'Gross_USD'] = movie_dict.get('Gross_USD')
    df.at[0,'No_of_Votes'] = movie_dict.get('No_of_Votes')
    df.at[0,'Certificate'] = movie_dict.get('Certificate')
    df.at[0,'Overview'] = movie_dict.get('Overview','')
    
    # Initialize genres
    for col in genre_cols:
        df.at[0,col] = 0
    
    genres = movie_dict.get('Genre', [])
    if isinstance(genres, str):
        genres = [genres]
    for g in genres:
        if g in genre_cols:
            df.at[0,g] = 1
    
    return round(model.predict(df)[0],2)
```


## 8. Prediction Example

```python
new_movie = {
 'Series_Title': 'The Shawshank Redemption',
 'Released_Year': 1994,
 'Certificate': 'A',
 'Runtime_Min': 142,
 'Genre': 'Drama',
 'Overview': 'Two imprisoned men bond over a number of years...',
 'Meta_score': 80.0,
 'No_of_Votes': 2343110,
 'Gross_USD': 28341469
}

predicted_rating = predict_imdb_rating(new_movie, model, genre_cols)
print("Predicted IMDB Rating:", predicted_rating)
```

**Output:**

```
Predicted IMDB Rating: 8.61
```


## 9. Conclusion

* The model predicts IMDB ratings with an RMSE of 0.208.
* Gradient Boosting Regressor effectively captures nonlinear interactions between numeric, categorical, text, and genre features.
* The pipeline is ready to handle new movie data, providing reliable rating estimates for decision-making in movie development.

