# Projeto_Reconhecimento_Digitos.py
# Este script implementa o reconhecimento de dígitos utilizando modelos de aprendizado de máquina

# %% [markdown]
# ## Importação de Bibliotecas

# %%
import numpy as np
import pandas as pd
from src.preprocessing import preprocess_images, filter_digits
from src.classifiers import train_perceptron, train_linear_regression, train_logistic_regression, predict
from src.evaluation import confusion_matrix, classification_report
from src.plotting import plot_data, plot_decision_boundary, plot_confusion_matrix
from src.utils import load_data, save_data

# %% [markdown]
# ## 1. Pré-processamento dos Dados
# - Nesta etapa, processamos os dados originais do MNIST, extraindo as colunas de **intensidade** e **simetria**.
# - Geramos os arquivos `train_redu.csv` e `test_redu.csv` com os dados reduzidos.

# %%
# Arquivos de entrada e saída
train_file = 'data/train.csv'
test_file = 'data/test.csv'
train_redu_file = 'data/train_redu.csv'
test_redu_file = 'data/test_redu.csv'

# Pré-processamento: cálculo de intensidade e simetria
preprocess_images(train_file, train_redu_file)
preprocess_images(test_file, test_redu_file)

# %% [markdown]
# ## 2. Classificação Binária (Dígitos 1 x 5)
# - Nesta seção, realizamos a classificação binária entre os dígitos 1 e 5 utilizando três modelos lineares:
#   - **Perceptron**
#   - **Regressão Linear**
#   - **Regressão Logística**
# - Vamos treinar cada modelo e avaliar suas performances.

# %%
# Filtrar dados para dígitos 1 e 5
train1x5_file = 'data/train1x5.csv'
test1x5_file = 'data/test1x5.csv'
filter_digits(train_redu_file, train1x5_file, digit1=1, digit2=5)
filter_digits(test_redu_file, test1x5_file, digit1=1, digit2=5)

# Carregar dados filtrados
train1x5_data = load_data(train1x5_file)
test1x5_data = load_data(test1x5_file)

# Separar características e rótulos
X_train = train1x5_data[['intensidade', 'simetria']].to_numpy()
y_train = train1x5_data['label'].replace({1: +1, 5: -1}).to_numpy()

X_test = test1x5_data[['intensidade', 'simetria']].to_numpy()
y_test = test1x5_data['label'].replace({1: +1, 5: -1}).to_numpy()

# %% [markdown]
# ### 2.1 Treinamento dos Modelos
# Vamos treinar os três classificadores lineares: **Perceptron**, **Regressão Linear** e **Regressão Logística**.

# %%
# Treinar o Perceptron
model_perceptron = train_perceptron(X_train, y_train)

# Treinar a Regressão Linear
model_linear = train_linear_regression(X_train, y_train)

# Treinar a Regressão Logística
model_logistic = train_logistic_regression(X_train, y_train)

# %% [markdown]
# ### 2.2 Avaliação dos Modelos
# - Realizamos previsões com os modelos treinados e avaliamos o desempenho utilizando a **matriz de confusão** e o **relatório de classificação**.

# %%
# Predições com os modelos
y_pred_perceptron = predict(model_perceptron, X_test)
y_pred_linear = predict(model_linear, X_test)
y_pred_logistic = predict(model_logistic, X_test)

# %% [markdown]
# ### 2.3 Métricas de Avaliação
# Avaliamos o desempenho dos três classificadores com base na matriz de confusão e nas métricas de classificação.

# %%
# Avaliar o Perceptron
print("Perceptron - Matriz de Confusão:")
conf_matrix_perceptron = confusion_matrix(y_test, y_pred_perceptron)
print(conf_matrix_perceptron)
print("\nRelatório de Classificação Perceptron:")
print(classification_report(y_test, y_pred_perceptron))

# Avaliar a Regressão Linear
print("\nRegressão Linear - Matriz de Confusão:")
conf_matrix_linear = confusion_matrix(y_test, y_pred_linear)
print(conf_matrix_linear)
print("\nRelatório de Classificação Regressão Linear:")
print(classification_report(y_test, y_pred_linear))

# Avaliar a Regressão Logística
print("\nRegressão Logística - Matriz de Confusão:")
conf_matrix_logistic = confusion_matrix(y_test, y_pred_logistic)
print(conf_matrix_logistic)
print("\nRelatório de Classificação Regressão Logística:")
print(classification_report(y_test, y_pred_logistic))

# %% [markdown]
# ### 2.4 Visualização dos Resultados
# - Vamos plotar os dados e as fronteiras de decisão dos três classificadores sobre o gráfico de **intensidade vs. simetria**.

# %%
# Plotar os dados
plot_data(X_train, y_train)

# Plotar as fronteiras de decisão para cada classificador
print("\nFronteira de Decisão - Perceptron:")
plot_decision_boundary(model_perceptron, X_train, y_train)

print("\nFronteira de Decisão - Regressão Linear:")
plot_decision_boundary(model_linear, X_train, y_train)

print("\nFronteira de Decisão - Regressão Logística:")
plot_decision_boundary(model_logistic, X_train, y_train)

# %% [markdown]
# ## 3. Classificação Multiclasse (Dígitos 0, 1, 4 e 5)
# Agora, vamos implementar a classificação para quatro dígitos: 0, 1, 4 e 5, utilizando a estratégia de **um contra todos** para cada classificador.

# %%
# Carregar dados completos
train_data = load_data(train_redu_file)
test_data = load_data(test_redu_file)

# Separar características e rótulos
X_train_full = train_data[['intensidade', 'simetria']].to_numpy()
y_train_full = train_data['label'].to_numpy()

X_test_full = test_data[['intensidade', 'simetria']].to_numpy()
y_test_full = test_data['label'].to_numpy()

# %% [markdown]
# ### 3.1 Treinamento Multiclasse
# Vamos treinar os classificadores para cada dígito utilizando a estratégia de **um contra todos**.

# %%
# Treinamento para cada dígito: 0, 1, 4 e 5
classifiers = {}

for digit in [0, 1, 4, 5]:
    y_train_digit = np.where(y_train_full == digit, +1, -1)
    model = train_logistic_regression(X_train_full, y_train_digit)
    classifiers[digit] = model

# %% [markdown]
# ### 3.2 Predição Multiclasse
# Para cada amostra no conjunto de teste, utilizamos as previsões de cada classificador e determinamos a classe final.

# %%
# Função de predição para multiclasse
def predict_multiclass(classifiers: dict, X: np.ndarray) -> np.ndarray:
    predictions = np.zeros((X.shape[0], len(classifiers)))
    for idx, digit in enumerate(classifiers):
        predictions[:, idx] = predict(classifiers[digit], X)
    
    return np.argmax(predictions, axis=1)

# Predição no conjunto de teste
y_pred_multiclass = predict_multiclass(classifiers, X_test_full)

# %% [markdown]
# ### 3.3 Avaliação Multiclasse
# Avaliamos a classificação multiclasse utilizando a **matriz de confusão** e as métricas de eficácia.

# %%
# Avaliar a classificação multiclasse
print("\nClassificação Multiclasse - Matriz de Confusão:")
conf_matrix_multiclass = confusion_matrix(y_test_full, y_pred_multiclass)
print(conf_matrix_multiclass)

print("\nRelatório de Classificação Multiclasse:")
print(classification_report(y_test_full, y_pred_multiclass))

# %% [markdown]
# ### 3.4 Visualização da Classificação Multiclasse
# Plotamos as retas de decisão de cada classificador para os quatro dígitos.

# %%
# Plotar a fronteira de decisão de cada dígito
for digit in classifiers:
    print(f"\nFronteira de Decisão para o dígito {digit}:")
    plot_decision_boundary(classifiers[digit], X_train_full, y_train_full)

# %% [markdown]
# ## 4. Comparação de Classificadores
# Por fim, comparamos os três classificadores em termos de acurácia, precisão, recall e F1-score.

# %%
# Comparação de métricas
# (As métricas já foram impressas anteriormente para cada classificador)
