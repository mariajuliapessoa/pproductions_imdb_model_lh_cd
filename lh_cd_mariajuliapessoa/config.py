import os

# Diretório base do projeto
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminhos dos dados
DATA_RAW = os.path.join(BASE_DIR, "data", "raw", "imdb_top_1000.csv")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed", "imdb_clean.csv")

# Caminho para relatórios
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

# Parâmetros do modelo
RANDOM_STATE = 42
TEST_SIZE = 0.2
