import pandas as pd
from typing import Union

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carrega os dados de um arquivo CSV e os retorna como um DataFrame.
    
    Args:
    - file_path: Caminho para o arquivo CSV
    
    Retorna:
    - Um DataFrame contendo os dados do arquivo CSV
    """
    return pd.read_csv(file_path)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Salva o DataFrame no arquivo especificado.
    
    Args:
    - data: DataFrame a ser salvo
    - file_path: Caminho onde o arquivo CSV será salvo
    """
    data.to_csv(file_path, index=False)
