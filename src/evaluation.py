from typing import Dict, Tuple
import numpy as np

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Gera a matriz de confusão entre os rótulos verdadeiros (y_true) e as previsões (y_pred).
    
    Args:
    - y_true: Vetor com os rótulos verdadeiros
    - y_pred: Vetor com os rótulos preditos pelo classificador
    
    Retorna:
    - Matriz de confusão: [2x2] para problemas binários ou [nxn] para multiclasse
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    n_labels = len(unique_labels)
    
    # Criar uma matriz de confusão com base nos rótulos únicos
    conf_matrix = np.zeros((n_labels, n_labels), dtype=int)
    
    for i, label_true in enumerate(unique_labels):
        for j, label_pred in enumerate(unique_labels):
            conf_matrix[i, j] = np.sum((y_true == label_true) & (y_pred == label_pred))
    
    return conf_matrix

def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Gera um relatório de eficácia de classificação contendo acurácia, precisão, recall e F1-score.
    
    Args:
    - y_true: Vetor com os rótulos verdadeiros
    - y_pred: Vetor com os rótulos preditos pelo classificador
    
    Retorna:
    - Dicionário com métricas de acurácia, precisão, recall e F1-score
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    # Inicializar as variáveis de métrica
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    precision_list = []
    recall_list = []
    f1_list = []
    
    for i in range(conf_matrix.shape[0]):
        tp = conf_matrix[i, i]  # True Positives
        fp = np.sum(conf_matrix[:, i]) - tp  # False Positives
        fn = np.sum(conf_matrix[i, :]) - tp  # False Negatives
        tn = np.sum(conf_matrix) - (tp + fp + fn)  # True Negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    # Cálculo das métricas macro (média das classes)
    precision_macro = np.mean(precision_list)
    recall_macro = np.mean(recall_list)
    f1_macro = np.mean(f1_list)
    
    return {
        'accuracy': accuracy,
        'precision': precision_macro,
        'recall': recall_macro,
        'f1_score': f1_macro
    }
