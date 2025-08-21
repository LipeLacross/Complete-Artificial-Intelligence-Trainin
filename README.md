# Complete-Artificial-Intelligence-Trainin

## 🌐 [English Version of README](README_EN.md)

# Formação Completa em Inteligência Artificial - 2025

Repositório para o curso **Formação Completa em Inteligência Artificial** do professor **Fernando Amaral**, cobrindo **Machine Learning, Deep Learning, LLMs, IA Generativa, NLP, Agentes, Visão Computacional, Detecção de Anomalias, Algoritmos Genéticos, Lógica Difusa** e muito mais.
Inclui teoria e prática com Python, implementações, projetos e materiais de apoio.

---

## 🔨 Funcionalidades do Projeto

* **Fundamentos Python** – exercícios introdutórios em scripts `.py`.
* **Algoritmos Clássicos de ML** – regressão, classificação, clustering, avaliação de modelos.
* **Machine Learning Avançado** – AutoML, seleção de atributos, desbalanceamento de classes.
* **Redes Neurais e Deep Learning** – MLP, CNN, LSTM, Autoencoders, Visão Computacional.
* **LLMs e IA Generativa** – GPT, Gemini, DeepSeek, Whisper, Stable Diffusion, DALL·E.
* **Processamento de Linguagem Natural (NLP)** – NLTK, embeddings, classificação de texto.
* **Agentes de IA** – agentes especializados e com RAG (LangChain).
* **Detecção de Anomalias** – Z-Score, IQR, Isolation Forest, Autoencoders, LSTM, ARIMA.
* **Algoritmos Genéticos e Otimização** – problemas binários e contínuos, simulated annealing.
* **Lógica Difusa (Fuzzy)** – criação de regras e sistemas de inferência.
* **Projeto Final (Adult Income)**:

  * Pipeline de ML completo com pré-processamento, SMOTE e seleção de atributos.
  * *Model selection* com **Optuna** (LogReg, RF, LGBM, XGBoost).
  * Avaliação final em `test.csv` com **F1 Score**.
  * Relatórios, gráficos e explicabilidade via **SHAP**.

---

### 📸 Exemplo Visual do Projeto

<div align="center">
  <img src="14. Projeto Final/reports/confusion_matrix.png" alt="Confusion Matrix - Projeto Final" width="80%" style="margin: 16px 0; border-radius: 10px;">
  <img src="14. Projeto Final/reports/shap_summary.png" alt="SHAP Summary - Projeto Final" width="80%" style="margin: 16px 0; border-radius: 10px;">
</div>

---

## ✔️ Técnicas e Tecnologias Utilizadas

* **Linguagem**: Python 3.12+
* **Bibliotecas principais**:
  `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `optuna`,
  `matplotlib`, `plotly`, `shap`, `lightgbm`, `xgboost`,
  `tensorflow/keras`, `torch`, `transformers`, `langchain`
* **Ambientes de execução**: Jupyter Notebook, Google Colab, VS Code
* **Gerenciamento de pacotes**: `pip`, `venv`
* **Artefatos de modelo**: `joblib`

---

## 📁 Estrutura do Projeto

* **3. Algoritmos de Machine Learning/**
  Regressão linear, Naive Bayes, Árvores de decisão, Random Forest, KNN, Clusters.

* **4. Tópicos Avançados de ML/**
  AutoML, seleção de atributos, PCA, dados desbalanceados.

* **5. Redes Neurais, Deep Learning e CV/**
  CNN, LSTM, MLP, Autoencoders, detecção de objetos (OpenCV).

* **6. Machine Learning Explicável/**
  Interpretação com LIME, SHAP, ELI5 e Interpret.

* **7. Processamento de Linguagem Natural/**
  NLP com NLTK, embeddings, classificação de texto.

* **8. LLMs e IA Generativa/**
  GPT, Gemini, DeepSeek, Whisper, Stable Diffusion, DALL·E.

* **9. Agentes de IA/**
  Agentes especializados, RAG e LangChain.

* **10. Detecção de Anomalias/**
  Z-Score, IQR, LOF, Isolation Forest, Autoencoders, LSTM, ARIMA.

* **11. Algoritmos Genéticos/**
  Representações binárias, valores reais, fitness functions.

* **12. Busca e Otimização/**
  Simulated Annealing.

* **13. Lógica Difusa/**
  Modelos fuzzy, criação de regras e inferência.

* **14. Projeto Final/**

  * `src/` – `train.py`, `evaluate.py`, `utils.py`
  * `data/` – `train.csv`, `validation.csv`, `test.csv`
  * `reports/` – métricas, gráficos, SHAP
  * `artifacts/` – pipelines e modelos salvos (`.joblib`)
  * `requirements.txt` – dependências com *pins* compatíveis

* **15-16. Fundamentos Python/**
  Exercícios simples em scripts:

  * `exercicio1.py` – amplitude de lista
  * `exercicio2.py` – imprime string na vertical
  * `exercicio3.py` – preço de transporte por peso

---

## 🛠️ Abrir e rodar o projeto

### 1. Preparar o ambiente Python (Projeto Final)

Certifique-se de ter **Python 3.12** instalado.

> Atenção: `numba`/`shap` não suportam Python 3.13 no Windows.

```bash
python -V
```

Crie e ative o ambiente:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

Instale dependências:

```bash
pip install --upgrade pip
pip install -r "14. Projeto Final/requirements.txt"
```

### 2. Treinar e avaliar o Projeto Final

```bash
python "14. Projeto Final/src/train.py" --n_trials 30
python "14. Projeto Final/src/evaluate.py"
```

Relatórios estarão em **14. Projeto Final/reports/**.

### 3. Executar exercícios básicos

```bash
python "15-16. Fundamentos Python/exercicio1.py"
python "15-16. Fundamentos Python/exercicio2.py"
python "15-16. Fundamentos Python/exercicio3.py"
```

### 4. Rodar notebooks

Abra os `.ipynb` no **Jupyter Notebook** ou **Google Colab**.

---

## 🌐 Deploy

### 🔹 Streamlit (demo interativa)

```bash
streamlit run app.py
```

Deploy em **Streamlit Cloud**.

### 🔹 FastAPI (API de predição)

```bash
uvicorn api:app --reload
```

Pode ser containerizado e hospedado em Railway/Fly.io/Heroku.

