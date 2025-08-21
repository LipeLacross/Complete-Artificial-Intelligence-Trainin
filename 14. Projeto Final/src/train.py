#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import f1_score

from utils import (
    TARGET_COL,
    POS_LABEL,
    build_preprocessor,
    compute_metrics,
    explain_with_shap,
    get_feature_types,
    get_model_definitions,
    load_data,
    make_training_pipeline,
    save_artifacts,
    save_json,
    save_text,
    set_seed,
    validate_columns_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Treino e Tuning - Adult Income")
    ap.add_argument("--train", default="data/train.csv", type=str, help="Caminho para train.csv")
    ap.add_argument("--val", default="data/validation.csv", type=str, help="Caminho para validation.csv")
    ap.add_argument("--reports_dir", default="reports", type=str, help="Pasta para relatórios")
    ap.add_argument("--n_trials", default=30, type=int, help="Número de trials no Optuna")
    ap.add_argument("--seed", default=42, type=int, help="Seed global")
    return ap.parse_args()


def build_objective(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    num_cols,
    cat_cols,
):
    models = get_model_definitions()

    def objective(trial: optuna.Trial) -> float:
        model_key = trial.suggest_categorical("model", list(models.keys()))
        model_def = models[model_key]

        # Começa com uma cópia "limpa" e sobrescreve chaves tunadas
        init = model_def["init"].copy()

        if model_key == "logreg":
            init["C"] = trial.suggest_float("C", 0.01, 10.0, log=True)

        elif model_key == "rf":
            init["n_estimators"] = trial.suggest_int("n_estimators", 200, 800, step=100)
            init["max_depth"] = trial.suggest_int("max_depth", 6, 30)
            init["min_samples_split"] = trial.suggest_int("min_samples_split", 2, 10)
            init["min_samples_leaf"] = trial.suggest_int("min_samples_leaf", 1, 5)

        elif model_key == "xgb":
            max_depth = trial.suggest_int("max_depth", 3, 10)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            n_estimators = trial.suggest_int("n_estimators", 200, 800, step=100)

            init["max_depth"] = max_depth
            init["learning_rate"] = learning_rate
            init["subsample"] = subsample
            init["colsample_bytree"] = colsample_bytree
            init["n_estimators"] = n_estimators
            init["scale_pos_weight"] = _scale_pos_weight(y_train)

        else:  # lgbm
            num_leaves = trial.suggest_int("num_leaves", 15, 255)
            max_depth = trial.suggest_int("max_depth", -1, 30)
            learning_rate = trial.suggest_float("learning_rate", 0.01, 0.2, log=True)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            n_estimators = trial.suggest_int("n_estimators", 300, 1200, step=100)

            init["num_leaves"] = num_leaves
            init["max_depth"] = max_depth
            init["learning_rate"] = learning_rate
            init["subsample"] = subsample
            init["colsample_bytree"] = colsample_bytree
            init["n_estimators"] = n_estimators

        ModelCls = model_def["cls"]
        model = ModelCls(**init)
        pre = build_preprocessor(num_cols, cat_cols)
        pipe = make_training_pipeline(pre, model)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_val)
        f1 = f1_score(y_val, preds, pos_label=POS_LABEL)
        trial.set_user_attr("f1_val", float(f1))
        return f1

    return objective


def _scale_pos_weight(y: pd.Series) -> float:
    values, counts = np.unique(y, return_counts=True)
    label_to_count = dict(zip(values, counts))
    pos = label_to_count.get(POS_LABEL, 1)
    neg = y.size - pos
    return max(1.0, float(neg) / float(pos))


def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.reports_dir, exist_ok=True)

    df_train = load_data(args.train)
    df_val = load_data(args.val)
    validate_columns_match(df_train, df_val)

    X_train = df_train.drop(columns=[TARGET_COL])
    y_train = df_train[TARGET_COL]
    X_val = df_val.drop(columns=[TARGET_COL])
    y_val = df_val[TARGET_COL]

    num_cols, cat_cols = get_feature_types(df_train, TARGET_COL)
    logging.info(f"Numéricas: {num_cols}")
    logging.info(f"Categorias: {cat_cols}")

    study = optuna.create_study(direction="maximize", study_name="adult_income_f1")
    objective = build_objective(X_train, y_train, X_val, y_val, num_cols, cat_cols)
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=False)

    trials_hist = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "f1_val": t.user_attrs.get("f1_val"),
        }
        for t in study.trials
        if t.value is not None
    ]
    hist_path = os.path.join(args.reports_dir, "tuning_history.json")
    save_json(trials_hist, hist_path)
    logging.info(f"Histórico de tuning salvo em {hist_path}")

    best_params = study.best_trial.params
    best_model_key = best_params.pop("model")
    model_defs = get_model_definitions()
    ModelCls = model_defs[best_model_key]["cls"]

    init = model_defs[best_model_key]["init"].copy()
    init.update(best_params)
    if best_model_key == "xgb":
        init["scale_pos_weight"] = _scale_pos_weight(pd.concat([y_train, y_val], axis=0))

    best_model = ModelCls(**init)

    pre = build_preprocessor(num_cols, cat_cols)
    pipe = make_training_pipeline(pre, best_model)

    X_all = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_all = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    logging.info(f"Treinando modelo final ({best_model_key}) em train+validation...")
    pipe.fit(X_all, y_all)

    val_preds = pipe.predict(X_val)
    val_proba = None
    try:
        val_proba = pipe.predict_proba(X_val)
    except Exception:
        pass
    val_metrics = compute_metrics(y_val, val_preds, val_proba)
    save_json(val_metrics, os.path.join(args.reports_dir, "validation_metrics.json"))
    save_text(
        f"Melhor modelo: {best_model_key}\nParams: {init}\nF1 Validation: {val_metrics.get('f1'):.4f}\n",
        os.path.join(args.reports_dir, "model_selection.txt"),
    )

    save_artifacts(pre, best_model, pipe, artifacts_dir="artifacts")

    sample = X_all.sample(min(200, len(X_all)), random_state=42)
    shap_path = explain_with_shap(pipe, sample, feature_names=X_all.columns.tolist(), out_dir="reports")
    if shap_path:
        logging.info(f"Figura SHAP salva em {shap_path}")

    logging.info("Treino concluído com sucesso.")


if __name__ == "__main__":
    main()
