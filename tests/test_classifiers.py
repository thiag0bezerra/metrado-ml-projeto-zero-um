import pytest
import numpy as np
from src.classifiers import train_perceptron, train_linear_regression, train_logistic_regression, predict

def test_perceptron_training() -> None:
    """
    Testa o treinamento do Perceptron.
    """
    # Dados de treino (simulados)
    X_train = np.array([[1, 1], [2, 2], [1, -1], [-2, -2], [-1, -1]])
    y_train = np.array([+1, +1, -1, -1, -1])
    
    # Treinar o Perceptron
    weights = train_perceptron(X_train, y_train)
    
    # Verificar se as previsões estão corretas
    predictions = predict(weights, X_train)
    assert np.array_equal(predictions, y_train)

def test_linear_regression_training() -> None:
    """
    Testa o treinamento da Regressão Linear.
    """
    # Dados de treino (simulados)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([+1, +1, -1, -1])
    
    # Treinar a Regressão Linear
    weights = train_linear_regression(X_train, y_train)
    
    # Verificar se o modelo ajusta corretamente os dados de treino
    predictions = predict(weights, X_train)
    expected_predictions = np.array([+1, +1, -1, -1])
    assert np.array_equal(predictions, expected_predictions)

def test_logistic_regression_training() -> None:
    """
    Testa o treinamento da Regressão Logística.
    """
    # Dados de treino (simulados)
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([+1, +1, -1, -1])
    
    # Treinar a Regressão Logística
    weights = train_logistic_regression(X_train, y_train)
    
    # Verificar se o modelo ajusta corretamente os dados de treino
    predictions = predict(weights, X_train)
    expected_predictions = np.array([+1, +1, -1, -1])
    assert np.array_equal(predictions, expected_predictions)
