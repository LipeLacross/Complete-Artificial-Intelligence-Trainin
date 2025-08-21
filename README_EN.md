# Complete-Artificial-Intelligence-Trainin

## 🌐 [Versão em Português do README](README.md)

# Complete Artificial Intelligence Training – 2025

Repository for the **Complete Artificial Intelligence Training** course by **Fernando Amaral**, covering **Machine Learning, Deep Learning, LLMs, Generative AI, NLP, Agents, Computer Vision, Anomaly Detection, Genetic Algorithms, Fuzzy Logic**, and much more.
It includes theory and practice with Python, implementations, projects, and support materials.

---

## 🔨 Project Features

* **Python Fundamentals** – introductory exercises in `.py` scripts.
* **Classic ML Algorithms** – regression, classification, clustering, model evaluation.
* **Advanced Machine Learning** – AutoML, feature selection, class imbalance handling.
* **Neural Networks and Deep Learning** – MLP, CNN, LSTM, Autoencoders, Computer Vision.
* **LLMs and Generative AI** – GPT, Gemini, DeepSeek, Whisper, Stable Diffusion, DALL·E.
* **Natural Language Processing (NLP)** – NLTK, embeddings, text classification.
* **AI Agents** – specialized agents with RAG (LangChain).
* **Anomaly Detection** – Z-Score, IQR, Isolation Forest, Autoencoders, LSTM, ARIMA.
* **Genetic Algorithms and Optimization** – binary and continuous problems, simulated annealing.
* **Fuzzy Logic** – rule-based systems and inference.
* **Final Project (Adult Income)**:

  * Full ML pipeline with preprocessing, SMOTE, and feature selection.
  * *Model selection* with **Optuna** (LogReg, RF, LGBM, XGBoost).
  * Final evaluation on `test.csv` with **F1 Score**.
  * Reports, plots, and explainability via **SHAP**.

---

### 📸 Project Visual Example

<div align="center">
  <img src="14. Projeto Final/reports/confusion_matrix.png" alt="Confusion Matrix - Final Project" width="80%" style="margin: 16px 0; border-radius: 10px;">
  <img src="14. Projeto Final/reports/shap_summary.png" alt="SHAP Summary - Final Project" width="80%" style="margin: 16px 0; border-radius: 10px;">
</div>

---

## ✔️ Techniques and Technologies Used

* **Language**: Python 3.12+
* **Core Libraries**:
  `pandas`, `numpy`, `scikit-learn`, `imbalanced-learn`, `optuna`,
  `matplotlib`, `plotly`, `shap`, `lightgbm`, `xgboost`,
  `tensorflow/keras`, `torch`, `transformers`, `langchain`
* **Execution Environments**: Jupyter Notebook, Google Colab, VS Code
* **Package Management**: `pip`, `venv`
* **Model Artifacts**: `joblib`

---

## 📁 Project Structure

* **3. Machine Learning Algorithms/**
  Linear Regression, Naive Bayes, Decision Trees, Random Forest, KNN, Clustering.

* **4. Advanced ML Topics/**
  AutoML, feature selection, PCA, imbalanced datasets.

* **5. Neural Networks, Deep Learning & CV/**
  CNN, LSTM, MLP, Autoencoders, object detection (OpenCV).

* **6. Explainable ML/**
  Model interpretation with LIME, SHAP, ELI5, and Interpret.

* **7. Natural Language Processing/**
  NLP with NLTK, embeddings, text classification.

* **8. LLMs & Generative AI/**
  GPT, Gemini, DeepSeek, Whisper, Stable Diffusion, DALL·E.

* **9. AI Agents/**
  Specialized agents, RAG with LangChain.

* **10. Anomaly Detection/**
  Z-Score, IQR, LOF, Isolation Forest, Autoencoders, LSTM, ARIMA.

* **11. Genetic Algorithms/**
  Binary, real-valued encodings, fitness functions.

* **12. Search & Optimization/**
  Simulated Annealing.

* **13. Fuzzy Logic/**
  Fuzzy models, rule creation, and inference.

* **14. Final Project/**

  * `src/` – `train.py`, `evaluate.py`, `utils.py`
  * `data/` – `train.csv`, `validation.csv`, `test.csv`
  * `reports/` – metrics, plots, SHAP
  * `artifacts/` – saved pipelines and models (`.joblib`)
  * `requirements.txt` – dependencies with compatible version pins

* **15-16. Python Fundamentals/**
  Simple exercises in scripts:

  * `exercicio1.py` – list range
  * `exercicio2.py` – print string vertically
  * `exercicio3.py` – transport price by weight

---

## 🛠️ Running the Project

### 1. Prepare the Python environment (Final Project)

Make sure **Python 3.12** is installed.

> Note: `numba`/`shap` are not supported on Python 3.13 for Windows.

```bash
python -V
```

Create and activate the environment:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r "14. Projeto Final/requirements.txt"
```

### 2. Train and Evaluate the Final Project

```bash
python "14. Projeto Final/src/train.py" --n_trials 30
python "14. Projeto Final/src/evaluate.py"
```

Reports will be stored in **14. Projeto Final/reports/**.

### 3. Run Basic Exercises

```bash
python "15-16. Fundamentos Python/exercicio1.py"
python "15-16. Fundamentos Python/exercicio2.py"
python "15-16. Fundamentos Python/exercicio3.py"
```

### 4. Run Notebooks

Open `.ipynb` files in **Jupyter Notebook** or **Google Colab**.

---

## 🌐 Deployment

### 🔹 Streamlit (interactive demo)

```bash
streamlit run app.py
```

Deploy via **Streamlit Cloud**.

### 🔹 FastAPI (prediction API)

```bash
uvicorn api:app --reload
```

Can be containerized and hosted on Railway/Fly.io/Heroku.

