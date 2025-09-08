````markdown
# IMDB Rating Prediction – Desafio Cientista de Dados

## Project Overview

This project is part of the "Desafio Cientista de Dados" by Indicium, in collaboration with the PProductions studio.  
The goal is to analyze a movie dataset and build a model capable of predicting IMDB ratings based on movie characteristics such as runtime, genre, certificate, overview, votes, gross revenue, and meta scores.

The project includes:
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Predictive Modeling using Gradient Boosting Regressor
- Saving the trained model for future predictions

---

## Project Structure

```text
|-- LICENSE
|-- Makefile
|-- README.md
|-- data
  |-- processed
    |-- imdb_clean.csv
  |-- raw
    |-- desafio_indicium_imdb.csv
|-- dataset.py
|-- lh_cd_mariajuliapessoa
  |-- __init__.py
  |-- config.py
|-- models
  |-- imdb_rating_model.pkl
|-- notebooks
  |-- EDA.ipynb
  |-- Modeling.ipynb
|-- reports
  |-- analysis
    |-- eda_report.md
    |-- modeling_report.md
  |-- figures
|-- requirements.txt
|-- src
  |-- __init__.py
  |-- __pycache__
    |-- *.pyc
  |-- data_prep.py
  |-- imports.py
  |-- plots.py
````

---

## Folder and File Description

* **data/raw** – Original CSV dataset provided by Indicium.
* **data/processed** – Cleaned dataset ready for analysis (`imdb_clean.csv`).
* **notebooks** – Jupyter notebooks containing EDA and Modeling workflows.
* **notebooks/reports/figures** – Figures generated within notebooks.
* **reports/analysis** – Markdown reports summarizing EDA and modeling results.
* **reports/figures** – Visualizations exported from notebooks for reporting purposes.
* **models** – Serialized trained model (`.pkl`) for IMDB rating prediction.
* **src** – Python modules for reusable code:

  * `data_prep.py` – Data cleaning and preprocessing functions
  * `plots.py` – Functions for visualizations
  * `imports.py` – Centralized import management
* **lh\_cd\_mariajuliapessoa** – Configuration files for project-specific settings.
* **requirements.txt** – Python package dependencies.
* **Makefile / LICENSE** – Optional project utilities and license information.

---

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/<your-username>/lh_cd_mariajuliapessoa.git
cd lh_cd_mariajuliapessoa
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run notebooks:

* `EDA.ipynb` – Explore and visualize dataset.
* `Modeling.ipynb` – Train model, evaluate performance, and make predictions.

---

## Usage Example

```python
from src.data_prep import load_data
from src.analysis import some_analysis_function
from src.plots import plot_distribution

# Load processed data
df = load_data("data/processed/imdb_clean.csv")
```

Predict IMDB rating for a new movie:

```python
from src.model_predict import predict_imdb_rating
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
rating = predict_imdb_rating(new_movie)
print(rating)  # Output: 8.61
```

---

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

```
