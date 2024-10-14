import pytest
import numpy as np
from src.evaluation import confusion_matrix, classification_report

def test_confusion_matrix() -> None:
    """
    Testa a criação da matriz de confusão.
    """
    # Teste com um problema binário
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Matriz esperada: [[2, 1], [1, 4]]
    expected_matrix = np.array([[2, 1], [1, 4]])
    
    assert np.array_equal(conf_matrix, expected_matrix)

def test_classification_report() -> None:
    """
    Testa o cálculo das métricas de classificação.
    """
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    y_pred = np.array([1, 0, 1, 0, 0, 1, 1, 1])
    
    report = classification_report(y_true, y_pred)
    
    # Métricas esperadas (calculadas manualmente)
    expected_report = {
        'accuracy': 6 / 8,  # 6 acertos em 8 exemplos
        'precision': (4 / 5 + 2 / 3) / 2,  # Média das precisões
        'recall': (4 / 5 + 2 / 3) / 2,     # Média dos recalls
        'f1_score': (8 / 11 + 4 / 5) / 2   # Média do F1-score
    }
    
    # Verificar se as métricas estão corretas
    assert np.isclose(report['accuracy'], expected_report['accuracy'])
    assert np.isclose(report['precision'], expected_report['precision'])
    assert np.isclose(report['recall'], expected_report['recall'])
    assert np.isclose(report['f1_score'], expected_report['f1_score'])
