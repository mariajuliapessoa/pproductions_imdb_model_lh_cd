import pandas as pd
from lh_cd_mariajuliapessoa.config import data_raw

def load_raw_data(path=data_raw):
    return pd.read_csv(path)
