import pytest
import pandas as pd
import numpy as np
from src.preprocessing import calculate_intensity, calculate_symmetry

def test_calculate_intensity() -> None:
    """
    Testa o cálculo de intensidade de uma imagem simples.
    """
    # Criar uma imagem de exemplo (28x28 pixels)
    image = pd.Series(np.array([0, 255] * 392))  # Alternando entre 0 e 255

    # A intensidade esperada é (392 * 255) / 255 = 392.0
    intensity = calculate_intensity(image)
    assert intensity == 392.0

def test_calculate_symmetry() -> None:
    """
    Testa o cálculo de simetria de uma imagem simples.
    """
    # Criar uma imagem de exemplo (28x28 pixels) simétrica
    image = pd.Series(np.zeros(28 * 28))  # Todos os pixels são 0, simetria perfeita

    # Simetria esperada deve ser 0, já que a imagem é perfeitamente simétrica
    symmetry = calculate_symmetry(image)
    assert symmetry == 0.0

    # Testar com uma imagem assimétrica
    image_asymmetric = pd.Series(np.arange(28 * 28))
    symmetry_asymmetric = calculate_symmetry(image_asymmetric)
    
    # A simetria assimétrica não será 0, mas garantimos que seja calculada corretamente
    assert symmetry_asymmetric > 0

def test_preprocess_images(tmpdir) -> None:
    """
    Testa a função de pré-processamento de imagens.
    """
    # Criar um arquivo temporário com um dataset de teste
    test_data = pd.DataFrame({
        'label': [1, 5],
        **{f'pixel{i}': [0, 255] for i in range(784)}  # Duas imagens: uma toda branca, outra preta
    })
    input_file = tmpdir.join("test_input.csv")
    output_file = tmpdir.join("test_output.csv")
    test_data.to_csv(input_file, index=False)
    
    # Chamar a função de pré-processamento
    preprocess_images(input_file, output_file)
    
    # Verificar os resultados
    output_data = pd.read_csv(output_file)
    assert 'intensidade' in output_data.columns
    assert 'simetria' in output_data.columns

    # A primeira imagem tem intensidade 0, a segunda tem intensidade 784
    assert output_data['intensidade'].iloc[0] == 0.0
    assert output_data['intensidade'].iloc[1] == 784.0 / 255.0  # Tudo preto
