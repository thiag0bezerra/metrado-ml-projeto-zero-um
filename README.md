# Projeto de Reconhecimento de Dígitos

Este projeto implementa um sistema de reconhecimento de dígitos manuscritos utilizando técnicas de Aprendizado de Máquina, aplicadas no dataset adaptado do MNIST. O foco é a implementação de três classificadores lineares (Perceptron, Regressão Linear, e Regressão Logística) e a comparação de suas performances em classificações binárias e multiclasse.

## Estrutura do Projeto

O projeto segue uma estrutura modular organizada para facilitar a manutenção, execução e análise:

```
/projeto_reconhecimento_digitos/
│
├── data/
│   ├── train.csv                  # Dataset original de treino
│   ├── test.csv                   # Dataset original de teste
│   ├── train_redu.csv             # Dataset reduzido com intensidade e simetria
│   ├── test_redu.csv              # Dataset reduzido com intensidade e simetria
│   ├── train1x5.csv               # Dataset filtrado para a classificação 1x5
│   └── test1x5.csv                # Dataset filtrado para a classificação 1x5
│
├── notebooks/
│   ├── Projeto_Reconhecimento_Digitos.ipynb    # Notebook principal com código e saídas
│   ├── Projeto_Reconhecimento_Digitos.py    # Equivalente em script python
│
├── src/
│   ├── __init__.py                # Arquivo de inicialização do módulo Python
│   ├── preprocessing.py           # Funções de pré-processamento: cálculo de intensidade e simetria
│   ├── classifiers.py             # Implementação dos classificadores (Perceptron, Regressão Linear, Regressão Logística)
│   ├── evaluation.py              # Funções de avaliação: matriz de confusão, métricas (acurácia, precisão, recall, f1-score)
│   ├── plotting.py                # Funções para plotar gráficos e visualizações
│   └── utils.py                   # Funções auxiliares para carregar dados, salvar resultados, etc.
│
├── tests/
│   ├── __init__.py                # Arquivo de inicialização do módulo de testes
│   ├── test_preprocessing.py      # Testes unitários para a parte de pré-processamento
│   ├── test_classifiers.py        # Testes unitários para os classificadores
│   ├── test_evaluation.py         # Testes unitários para a avaliação
│
├── requirements.txt               # Bibliotecas necessárias (numpy, pandas, matplotlib, scikit-learn)
├── README.md                      # Instruções do projeto
└── run.sh                         # Script de execução automatizada (opcional)
```

### Descrição dos Componentes:

1. **`data/`**: Contém os datasets de treino e teste:
   - `train.csv` e `test.csv`: Datasets originais do MNIST adaptado.
   - `train_redu.csv` e `test_redu.csv`: Dados pré-processados com as colunas de intensidade e simetria.
   - `train1x5.csv` e `test1x5.csv`: Dados filtrados para a classificação binária entre os dígitos 1 e 5.

2. **`notebooks/`**: Notebook Jupyter contendo todo o código e visualizações para treinamento, avaliação e plotagem dos resultados.

3. **`src/`**: Código fonte principal organizado em módulos:
   - `preprocessing.py`: Funções para calcular a intensidade e simetria das imagens.
   - `classifiers.py`: Implementações dos classificadores Perceptron, Regressão Linear e Regressão Logística.
   - `evaluation.py`: Funções de avaliação que geram a matriz de confusão e métricas de desempenho.
   - `plotting.py`: Funções para criar gráficos para a visualização dos resultados.
   - `utils.py`: Funções auxiliares para manipulação de dados e resultados.

4. **`tests/`**: Testes unitários para garantir a funcionalidade correta dos módulos de pré-processamento, classificadores e avaliação.

5. **`requirements.txt`**: Lista das dependências necessárias para rodar o projeto.

### Como Executar o Projeto

1. **Instale as dependências**:
   ```
   pip install -r requirements.txt
   ```

2. **Rode o notebook Jupyter**:
   - Abra o notebook `Projeto_Reconhecimento_Digitos.ipynb` e execute as células para treinar e avaliar os classificadores.

3. **Testes Unitários**:
   - Rode os testes para verificar a implementação:
   ```
   python -m unittest discover -s tests
   ```

4. **Automação (opcional)**:
   - Use o script `run.sh` para rodar o pipeline completo (pré-processamento, treino e avaliação).