````markdown
# IMDB Rating Prediction – Desafio Cientista de Dados

## Project Overview

This project is part of the "Desafio Cientista de Dados" organized by **Indicium**, in collaboration with the **PProductions studio**.  
The main goal is to analyze a movie dataset and build a predictive model capable of estimating **IMDB ratings** based on movie attributes such as:

- Runtime
- Genre
- Certificate
- Overview
- Number of votes
- Gross revenue
- Meta score

The project workflow includes:

1. **Exploratory Data Analysis (EDA)** – Understand patterns, distributions, and correlations in the dataset.
2. **Feature Engineering** – Clean and transform raw data to create meaningful features for modeling.
3. **Predictive Modeling** – Train a **Gradient Boosting Regressor** model for rating prediction.
4. **Model Serialization** – Save the trained model for future inference.

---

## Project Structure

```text
lh_cd_mariajuliapessoa/
├── LICENSE
├── Makefile
├── README.md
├── data/
│   ├── raw/
│   │   └── desafio_indicium_imdb.csv       # Original dataset
│   └── processed/
│       └── imdb_clean.csv                  # Cleaned dataset for analysis
├── dataset.py                               # Data loading and preparation script
├── lh_cd_mariajuliapessoa/
│   ├── __init__.py
│   └── config.py                            # Project-specific configuration settings
├── models/
│   └── imdb_rating_model.pkl                # Trained model for IMDB rating prediction
├── notebooks/
│   ├── EDA.ipynb                            # Exploratory Data Analysis notebook
│   └── Modeling.ipynb                       # Modeling workflow notebook
├── requirements.txt                         # Python dependencies
└── src/
    ├── __init__.py
    ├── data_prep.py                         # Data cleaning and preprocessing functions
    ├── plots.py                             # Visualization functions
    └── imports.py                            # Centralized imports
````

---

## Folder and File Description

* **data/raw** – Contains the original CSV dataset from Indicium.
* **data/processed** – Contains the cleaned dataset (`imdb_clean.csv`) ready for analysis.
* **notebooks/** – Jupyter notebooks with exploratory analysis and model building steps:

  * `EDA.ipynb` – Analysis of distributions, correlations, missing values, and data insights.
  * `Modeling.ipynb` – Feature engineering, model training, evaluation, and prediction.
* **models/** – Serialized model file (`.pkl`) to be used for prediction.
* **src/** – Python modules for reusable functions:

  * `data_prep.py` – Functions for data cleaning and preprocessing.
  * `plots.py` – Functions for creating charts and visualizations.
  * `imports.py` – Centralized imports to standardize dependencies.
* **lh\_cd\_mariajuliapessoa/** – Project configuration files:

  * `config.py` – Paths, constants, and settings used across scripts and notebooks.
* **requirements.txt** – List of Python packages required for running the project.
* **Makefile / LICENSE** – Optional utilities and license information.

---

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/lh_cd_mariajuliapessoa.git
cd lh_cd_mariajuliapessoa
```

2. **Create and activate a virtual environment** (recommended)

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run notebooks**

* `EDA.ipynb` – Explore and visualize the dataset.
* `Modeling.ipynb` – Perform feature engineering, train the model, evaluate performance, and make predictions.

---

## Usage Example

### Loading the dataset

```python
from src.data_prep import load_data
from src.plots import plot_distribution

# Load processed data
df = load_data("data/processed/imdb_clean.csv")

# Example visualization
plot_distribution(df, "IMDB_Rating")
```

### Predicting IMDB rating for a new movie

```python
from src.model_predict import predict_imdb_rating

new_movie = {
    'Series_Title': 'Twilight',
    'Released_Year': 2008,
    'Certificate': 'U',
    'Runtime_Min': 130,
    'Genre': 'Romance',
    'Overview': 'High-school student Bella Swan, always a bit of a misfit...',
    'Meta_score': 56.0,
    'No_of_Votes': 19780000,
    'Gross_USD': 408526215
}

rating = predict_imdb_rating(new_movie)
print(f"Predicted IMDB rating: {rating}")
```

---

## Key Features

* Modular project structure for **reusability** and **scalability**.
* Cleaned dataset ready for further analysis or production use.
* Gradient Boosting model providing **robust IMDB rating predictions**.
* Easy-to-use prediction function for new movies.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

```
