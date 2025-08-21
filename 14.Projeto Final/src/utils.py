#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import logging
import os
import warnings
from typing import Dict, List, Tuple, Optional

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, LabelEncoder
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

TARGET_COL = "income"
POS_LABEL = ">50K"  # rótulo positivo para F1


# -------------------------
# Wrapper para codificar rótulos internamente (0/1) e devolver originais
# -------------------------
class LabelEncodedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator):
        self.base_estimator = base_estimator
        self.le_ = None
        self.est_ = None

    def fit(self, X, y):
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(np.asarray(y))
        self.est_ = clone(self.base_estimator)
        self.est_.fit(X, y_enc)
        return self

    def predict(self, X):
        y_enc = self.est_.predict(X)
        return self.le_.inverse_transform(y_enc)

    def predict_proba(self, X):
        if not hasattr(self.est_, "predict_proba"):
            raise AttributeError("Estimador base não suporta predict_proba")
        proba = self.est_.predict_proba(X)
        return proba  # ordem das colunas segue classes_ internas (0,1)

    def get_params(self, deep=True):
        # Não expomos os params internos para tuning; usamos só após escolher o modelo
        return {"base_estimator": self.base_estimator}

    def set_params(self, **params):
        if "base_estimator" in params:
            self.base_estimator = params["base_estimator"]
        return self


# -------------------------
# Utilidades de dados
# -------------------------
def set_seed(seed: int = 42) -> None:
    import random
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except Exception:
        pass


def _normalize_missing(df: pd.DataFrame) -> pd.DataFrame:
    # Converte "?" e strings em branco em NaN
    return df.replace(to_replace=[r"^\s*\?\s*$", r"^\s*$"], value=np.nan, regex=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_missing(df)
    return df


def get_feature_types(df: pd.DataFrame, target: str = TARGET_COL) -> Tuple[List[str], List[str]]:
    if target not in df.columns:
        raise ValueError(f"Coluna alvo '{target}' não encontrada em {list(df.columns)}")
    feature_df = df.drop(columns=[target])
    cat_cols = feature_df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    num_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    return num_cols, cat_cols


# -------------------------
# Pré-processamento
# -------------------------
def build_preprocessor(
    num_cols: List[str],
    cat_cols: List[str],
    impute_num_strategy: str = "median",
    impute_cat_strategy: str = "most_frequent",
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_num_strategy)),
            ("scaler", RobustScaler(with_centering=True, with_scaling=True)),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=impute_cat_strategy)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_cols),
            ("cat", categorical_pipeline, cat_cols),
        ],
        remainder="drop",
        n_jobs=None,
        verbose_feature_names_out=False,
    )
    return preprocessor


# -------------------------
# Modelos e defaults (sem chaves que vamos tunar)
# -------------------------
def get_model_definitions() -> Dict[str, Dict]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier

    return {
        "logreg": {
            "cls": LogisticRegression,
            "init": dict(
                solver="lbfgs",
                max_iter=1000,
                n_jobs=-1,
                class_weight="balanced",
            ),
        },
        "rf": {
            "cls": RandomForestClassifier,
            "init": dict(
                # n_estimators será tunado
                max_depth=None,
                class_weight="balanced",
                n_jobs=-1,
                random_state=42,
            ),
        },
        "xgb": {
            "cls": XGBClassifier,
            "init": dict(
                # n_estimators e max_depth serão tunados
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=42,
            ),
        },
        "lgbm": {
            "cls": LGBMClassifier,
            "init": dict(
                # n_estimators será tunado
                max_depth=-1,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="binary",
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
        },
    }


def make_training_pipeline(preprocessor: ColumnTransformer, model) -> ImbPipeline:
    """
    Pré -> Seleção de atributos (L1) -> SMOTE -> Modelo
    """
    selector = SelectFromModel(
        LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000, class_weight="balanced"),
        max_features=80,   # ajuste opcional
        threshold="median"
    )

    pipe = ImbPipeline(
        steps=[
            ("pre", preprocessor),
            ("fs", selector),
            ("smote", SMOTE(random_state=42)),
            ("clf", LabelEncodedClassifier(model)),
        ]
    )
    return pipe



# -------------------------
# Métricas e relatórios
# -------------------------
def compute_metrics(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    metrics = {
        "f1": f1_score(y_true, y_pred, pos_label=POS_LABEL),
        "precision": precision_score(y_true, y_pred, pos_label=POS_LABEL),
        "recall": recall_score(y_true, y_pred, pos_label=POS_LABEL),
    }
    if y_proba is not None:
        try:
            y_score = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
            metrics["roc_auc"] = roc_auc_score((np.array(y_true) == POS_LABEL).astype(int), y_score)
        except Exception:
            pass
    return metrics


def save_json(obj: Dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_text(text: str, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def plot_confusion(cm: np.ndarray, labels: List[str], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig = plt.figure(figsize=(4.5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Matriz de Confusão")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45, ha="right")
    plt.yticks(tick_marks, labels)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("Verdadeiro")
    plt.xlabel("Predito")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def model_is_tree_based(model) -> bool:
    name = model.__class__.__name__.lower()
    return any(k in name for k in ["xgb", "lgbm", "randomforest", "gradientboost", "extratrees"])


def explain_with_shap(
    fitted_pipeline: Pipeline | ImbPipeline,
    X_sample: pd.DataFrame,
    feature_names: List[str],
    out_dir: str,
) -> Optional[str]:
    try:
        import shap
    except Exception:
        logging.warning("pacote shap não disponível, pulando XAI.")
        return None

    os.makedirs(out_dir, exist_ok=True)

    try:
        model = fitted_pipeline.named_steps["clf"].est_
    except Exception:
        logging.warning("Não foi possível localizar estimador interno para SHAP.")
        return None

    X_trans = fitted_pipeline.named_steps["pre"].transform(X_sample)
    try:
        ohe = fitted_pipeline.named_steps["pre"].named_transformers_["cat"].named_steps["encoder"]
        cat_feature_names = ohe.get_feature_names_out()
    except Exception:
        cat_feature_names = []
    num_names = fitted_pipeline.named_steps["pre"].transformers_[0][2]
    names = list(num_names) + list(cat_feature_names)
    if len(names) != X_trans.shape[1]:
        names = feature_names

    shap_plot_path = os.path.join(out_dir, "shap_summary.png")
    try:
        if model_is_tree_based(model):
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict_proba, X_trans[:200])
        shap_values = explainer.shap_values(X_trans[:200])
        values = shap_values[1] if isinstance(shap_values, list) and len(shap_values) > 1 else shap_values
        plt.figure()
        shap.summary_plot(values, X_trans[:200], feature_names=names, show=False)
        plt.tight_layout()
        plt.savefig(shap_plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        return shap_plot_path
    except Exception as e:
        logging.warning(f"Falha ao gerar SHAP: {e}")
        return None


# -------------------------
# Salvamento de artefatos
# -------------------------
def save_artifacts(
    preprocessor: ColumnTransformer,
    model,
    fitted_pipeline: Pipeline | ImbPipeline,
    artifacts_dir: str = "artifacts",
) -> None:
    os.makedirs(artifacts_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(artifacts_dir, "preprocessing.joblib"))
    joblib.dump(model, os.path.join(artifacts_dir, "best_model.joblib"))
    joblib.dump(fitted_pipeline, os.path.join(artifacts_dir, "best_pipeline.joblib"))
    logging.info(f"Artefatos salvos em '{artifacts_dir}'.")


# -------------------------
# Validações rápidas
# -------------------------
def validate_columns_match(train_df: pd.DataFrame, other_df: pd.DataFrame) -> None:
    missing = set(train_df.columns) - set(other_df.columns)
    if missing:
        raise ValueError(f"Arquivo de dados faltando colunas do treino: {missing}")
