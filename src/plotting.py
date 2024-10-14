import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def plot_data(X: np.ndarray, y: np.ndarray) -> None:
    """
    Plota os dados de intensidade e simetria em um gráfico 2D.
    Os rótulos (y) determinam a cor dos pontos no gráfico.
    
    Args:
    - X: Matriz de dados, onde cada linha é uma amostra e contém as características (intensidade, simetria)
    - y: Vetor de rótulos (dígitos) correspondentes às amostras
    """
    plt.figure(figsize=(8, 6))
    
    for digit in np.unique(y):
        indices = y == digit
        plt.scatter(X[indices, 0], X[indices, 1], label=f'Dígito {digit}', alpha=0.6)
    
    plt.xlabel('Intensidade')
    plt.ylabel('Simetria')
    plt.legend()
    plt.title('Dados de Intensidade vs Simetria')
    plt.grid(True)
    plt.show()

def plot_decision_boundary(model: np.ndarray, X: np.ndarray, y: np.ndarray) -> None:
    """
    Plota a fronteira de decisão de um classificador treinado sobre os dados.
    
    Args:
    - model: Vetor de coeficientes do modelo treinado
    - X: Matriz de dados (intensidade, simetria)
    - y: Vetor de rótulos (dígitos)
    """
    plot_data(X, y)
    
    # Gerar uma malha de pontos para traçar a fronteira
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = np.dot(np.c_[xx.ravel(), yy.ravel()], model[:-1]) + model[-1]
    Z = Z.reshape(xx.shape)

    # Plotar a fronteira de decisão
    plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')
    plt.show()

def plot_confusion_matrix(matrix: np.ndarray) -> None:
    """
    Plota a matriz de confusão.
    
    Args:
    - matrix: Matriz de confusão
    """
    plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = np.arange(len(matrix))
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    
    plt.ylabel('Rótulo Verdadeiro')
    plt.xlabel('Rótulo Previsto')
    plt.show()
