from typing import Tuple
import pandas as pd

def calculate_intensity(image: pd.Series) -> float:
    """Calcula a intensidade de uma imagem a partir de seus pixels."""
    ...

def calculate_symmetry(image: pd.Series) -> float:
    """Calcula a simetria (vertical e horizontal) de uma imagem."""
    ...

def preprocess_images(input_file: str, output_file: str) -> None:
    """
    Processa o arquivo CSV de entrada, calcula intensidade e simetria
    para cada imagem, e salva no arquivo de saída.
    """
    ...

def filter_digits(input_file: str, output_file: str, digit1: int, digit2: int) -> None:
    """
    Filtra os dígitos especificados em 'digit1' e 'digit2' de um arquivo CSV,
    e salva os dados filtrados no arquivo de saída.
    """
    ...
