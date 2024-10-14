from typing import Tuple, Dict
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gera a matriz de confusão entre os rótulos verdadeiros (y_true) e as previsões (y_pred).
    """
    ...

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Gera um relatório de eficácia de classificação contendo acurácia, precisão, recall e F1-score.
    Retorna um dicionário com as métricas.
    """
    ...
