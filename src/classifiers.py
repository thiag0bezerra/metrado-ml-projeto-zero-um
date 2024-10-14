import numpy as np
from typing import Tuple

def train_perceptron(X: np.ndarray, y: np.ndarray, max_iter: int = 1000, learning_rate: float = 0.01) -> np.ndarray:
    """
    Treina um modelo Perceptron usando o algoritmo de aprendizagem online.
    
    Args:
    - X: Matriz de dados de entrada (intensidade, simetria) [n_amostras, 2]
    - y: Vetor de rótulos alvo (+1 ou -1) [n_amostras]
    - max_iter: Número máximo de iterações para treinamento
    - learning_rate: Taxa de aprendizado
    
    Retorna:
    - Pesos ajustados (coeficientes do modelo) [3] (pesos + bias)
    """
    n_samples, n_features = X.shape
    # Inicializar pesos e bias (adicionando um peso extra para o bias)
    weights = np.zeros(n_features + 1)

    # Adicionar o termo de bias (x0 = 1) para todos os exemplos
    X_bias = np.c_[np.ones(n_samples), X]

    # Algoritmo de treinamento do Perceptron
    for _ in range(max_iter):
        for i in range(n_samples):
            if y[i] * np.dot(weights, X_bias[i]) <= 0:
                weights += learning_rate * y[i] * X_bias[i]

    return weights

def train_linear_regression(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Treina um modelo de Regressão Linear.
    
    Args:
    - X: Matriz de dados de entrada (intensidade, simetria) [n_amostras, 2]
    - y: Vetor de rótulos alvo (+1 ou -1) [n_amostras]
    
    Retorna:
    - Pesos ajustados (coeficientes do modelo) [3] (pesos + bias)
    """
    # Adicionar o termo de bias
    n_samples = X.shape[0]
    X_bias = np.c_[np.ones(n_samples), X]
    
    # Ajustar pesos pela fórmula da regressão linear: (X^T * X)^(-1) * X^T * y
    weights = np.linalg.inv(X_bias.T @ X_bias) @ X_bias.T @ y
    return weights

def train_logistic_regression(X: np.ndarray, y: np.ndarray, max_iter: int = 1000, learning_rate: float = 0.01) -> Tuple[np.ndarray, float]:
    """
    Treina um modelo de Regressão Logística usando gradiente descendente.
    
    Args:
    - X: Matriz de dados de entrada (intensidade, simetria) [n_amostras, 2]
    - y: Vetor de rótulos alvo (+1 ou -1) [n_amostras]
    - max_iter: Número máximo de iterações para treinamento
    - learning_rate: Taxa de aprendizado
    
    Retorna:
    - Pesos ajustados (coeficientes do modelo) [3] (pesos + bias)
    """
    n_samples, n_features = X.shape
    # Inicializar pesos e bias (adicionando um peso extra para o bias)
    weights = np.zeros(n_features + 1)

    # Adicionar o termo de bias (x0 = 1) para todos os exemplos
    X_bias = np.c_[np.ones(n_samples), X]

    # Função sigmoide
    def sigmoid(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-z))
    
    # Gradiente descendente
    for _ in range(max_iter):
        # Predição com a sigmoide
        y_pred = sigmoid(np.dot(X_bias, weights))
        
        # Gradiente para os pesos
        gradient = np.dot(X_bias.T, (y_pred - (y + 1) / 2)) / n_samples
        
        # Atualizar os pesos
        weights -= learning_rate * gradient

    return weights

def predict(model: np.ndarray, X: np.ndarray) -> np.ndarray:
    """
    Faz previsões com base no modelo treinado e nos dados X.
    
    Args:
    - model: Pesos ajustados do modelo (incluindo o bias)
    - X: Matriz de dados de entrada (intensidade, simetria) [n_amostras, 2]
    
    Retorna:
    - Vetor de previsões (+1 ou -1)
    """
    n_samples = X.shape[0]
    # Adicionar o termo de bias
    X_bias = np.c_[np.ones(n_samples), X]
    
    # Produto escalar entre os dados e os pesos
    predictions = np.dot(X_bias, model)
    
    # Predições finais: +1 se positivo, -1 se negativo
    return np.where(predictions >= 0, 1, -1)
