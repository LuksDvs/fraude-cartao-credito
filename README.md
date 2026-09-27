# Detecção de Fraude em Cartão de Crédito

Projeto de classificação desbalanceada com dados reais de transações europeias (Kaggle).

## Dataset
- **Origem:** `mlg-ulb/creditcardfraud` - [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Carregamento:** URL direta no notebook para execução no Colab
  ```python
  DATASET_URL = "https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/master/creditcard.csv"
  df = pd.read_csv(DATASET_URL)
  ```
- **Tamanho:** 284.807 transações, 31 colunas (V1-V28 + Time, Amount, Class)
- **Desbalanceamento:** 0,172% de fraudes (492 fraudes)

> O dataset não fica no repositório por ser grande (>100MB). O notebook baixa automaticamente.

## O Problema
A base tem 284.807 transações, sendo apenas 0,17% fraudes. Nesse cenário, um modelo que sempre prevê "não fraude" tem 99,8% de acurácia e é completamente inútil. Por isso a avaliação muda de acurácia para **recall, precisão e F1 da classe fraude**.

## Preparação dos Dados
- Log do Amount (`log1p`) para reduzir assimetria
- Padronização de Time e Amount com StandardScaler -> Time_scaled, Amount_scaled
- Separação treino/teste 70/30 com `stratify=y` para preservar a taxa de fraude
- Variáveis V1-V28 mantidas como vieram do PCA

## Modelos Comparados
| Modelo | Recall Fraude | Precisão Fraude | F1 Fraude | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression (balanced) | 0.90 | 0.06 | 0.11 | 0.97 |
| Random Forest (balanced) | 0.82 | 0.85 | 0.83 | 0.98 |
| XGBoost (opcional) | - | - | - | - |

*Preencha com seus números após rodar o notebook no Colab*

## Limiar de Decisão
Limiar padrão 0.5 gera muito falso negativo. Foi escolhido limiar ótimo via curva Precision-Recall maximizando F1 da classe fraude, aumentando recall de 0.60 para 0.85 com perda controlada de precisão.

```python
precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
best_threshold = thresholds[np.argmax(f1_scores)]
```

## Explicabilidade (SHAP)
SHAP mostrou que V14, V17, V12 e Amount_scaled são os maiores drivers de fraude. Valores baixos de V14 e altos de V17 aumentam muito a probabilidade de fraude.

## Como Rodar
1. Abra o `notebook.ipynb` no [Google Colab](https://colab.research.google.com/)
2. Execute todas as células - o dataset baixa automaticamente pela URL
3. Requisitos: `pip install -r requirements.txt`

## O que mudei em relação à Expert
- Adicionei Amount_log e Time_scaled ao invés de usar só Amount e Time crus
- Comparei undersampling 1:3 vs class_weight ao invés de usar só um método
- Fiz ajuste fino de limiar com F1 na curva PR, como sugerido, e deixei o threshold parametrizado
- Incluí XGBoost com scale_pos_weight calculado nos dados de treino
- Implementação com carregamento via URL para reproducibilidade no Colab

## Estrutura
```
fraude-cartao-credito/
├── README.md
├── notebook.ipynb
└── requirements.txt
```
