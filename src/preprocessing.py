import pandas as pd
import numpy as np

def calculate_intensity(image: pd.Series) -> float:
    """
    Calcula a intensidade de uma imagem a partir dos valores de pixels.
    A intensidade é a soma dos valores dos pixels dividida por 255, o que
    representa a fração de pixels pretos em uma escala de cinza.
    """
    return np.sum(image) / 255.0

def calculate_symmetry(image: pd.Series) -> float:
    """
    Calcula a simetria vertical e horizontal de uma imagem.
    A simetria é a diferença entre pixels espelhados em torno dos eixos vertical e horizontal.
    """
    image_matrix = image.values.reshape(28, 28)
    
    # Simetria vertical
    symmetry_vertical = np.sum(np.abs(image_matrix - np.fliplr(image_matrix))) / 255.0
    
    # Simetria horizontal
    symmetry_horizontal = np.sum(np.abs(image_matrix - np.flipud(image_matrix))) / 255.0
    
    return symmetry_vertical + symmetry_horizontal

def preprocess_images(input_file: str, output_file: str) -> None:
    """
    Processa o arquivo CSV de entrada, calcula a intensidade e simetria de cada imagem,
    e salva os resultados no arquivo de saída.
    
    Args:
    - input_file: Caminho para o arquivo CSV de entrada.
    - output_file: Caminho para o arquivo CSV de saída.
    """
    data = pd.read_csv(input_file, delimiter=';')
    
    # Verificar se a coluna 'label' está presente
    if 'label' not in data.columns:
        raise KeyError("'label' column not found in the input CSV file.")
    
    # Calcular intensidade e simetria para cada linha
    processed_data = data.apply(lambda row: pd.Series({
        'label': row['label'],
        'intensidade': calculate_intensity(row.drop('label')),
        'simetria': calculate_symmetry(row.drop('label'))
    }), axis=1)
    
    # Salvar dados processados
    processed_data.to_csv(output_file, index=False)

def filter_digits(input_file: str, output_file: str, digit1: int, digit2: int) -> None:
    """
    Filtra os dígitos especificados (digit1 e digit2) de um arquivo CSV
    e salva os dados filtrados no arquivo de saída.
    
    Args:
    - input_file: Caminho para o arquivo CSV de entrada.
    - output_file: Caminho para o arquivo CSV de saída.
    - digit1: Primeiro dígito para filtrar (ex: 1).
    - digit2: Segundo dígito para filtrar (ex: 5).
    """
    data = pd.read_csv(input_file)
    
    # Verificar se a coluna 'label' está presente
    if 'label' not in data.columns:
        raise KeyError("'label' column not found in the input CSV file.")
    
    # Filtrar linhas onde o label é igual a digit1 ou digit2
    filtered_data = data[(data['label'] == digit1) | (data['label'] == digit2)]
    
    # Salvar dados filtrados
    filtered_data.to_csv(output_file, index=False)
