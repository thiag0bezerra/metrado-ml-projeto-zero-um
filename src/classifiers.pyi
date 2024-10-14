from typing import Tuple
import numpy as np

def train_perceptron(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Treina o modelo Perceptron com os dados X e y.
    Retorna os coeficientes ajustados.
    """
    ...

def train_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Treina o modelo de Regressão Linear com os dados X e y.
    Retorna os coeficientes ajustados.
    """
    ...

def train_logistic_regression(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Treina o modelo de Regressão Logística com os dados X e y.
    Retorna os coeficientes ajustados e o valor de intercepto.
    """
    ...

def predict(model: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Faz previsões com base no modelo treinado e nos dados X.
    Retorna as previsões.
    """
    ...
