import numpy as np
import matplotlib.pyplot as plt

def plot_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plota os dados de intensidade e simetria em um gráfico 2D.
    Os rótulos (y) determinam a cor dos pontos no gráfico.
    """
    ...

def plot_decision_boundary(model: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    """
    Plota a fronteira de decisão de um classificador treinado sobre os dados.
    """
    ...

def plot_confusion_matrix(matrix: np.ndarray) -> None:
    """
    Plota a matriz de confusão.
    """
    ...
