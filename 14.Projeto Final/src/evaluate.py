#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from utils import (
    TARGET_COL,
    POS_LABEL,
    compute_metrics,
    explain_with_shap,
    load_data,
    plot_confusion,
    save_json,
    save_text,
    validate_columns_match,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    ap = argparse.ArgumentParser(description="Avaliação Final no Test Set")
    ap.add_argument("--test", default="data/test.csv", type=str, help="Caminho para test.csv")
    ap.add_argument("--artifacts_dir", default="artifacts", type=str, help="Pasta de artefatos")
    ap.add_argument("--reports_dir", default="reports", type=str, help="Pasta para relatórios")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.reports_dir, exist_ok=True)

    # Carrega dados
    df_test = load_data(args.test)
    X_test = df_test.drop(columns=[TARGET_COL])
    y_test = df_test[TARGET_COL]

    # Carrega artefatos
    pre_path = os.path.join(args.artifacts_dir, "preprocessing.joblib")
    model_path = os.path.join(args.artifacts_dir, "best_model.joblib")
    pipe_path = os.path.join(args.artifacts_dir, "best_pipeline.joblib")

    if os.path.exists(pipe_path):
        logging.info("Carregando pipeline completo...")
        pipeline = joblib.load(pipe_path)
    else:
        logging.info("Pipeline completo não encontrado; reconstruindo com preprocess + model.")
        pre = joblib.load(pre_path)
        model = joblib.load(model_path)
        from imblearn.pipeline import Pipeline as ImbPipeline
        from utils import make_training_pipeline

        pipeline = ImbPipeline(steps=[("pre", pre), ("clf", model)])

    # Predições
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)
    except Exception:
        pass

    # Métricas
    metrics = compute_metrics(y_test, y_pred, y_proba)
    save_json(metrics, os.path.join(args.reports_dir, "test_metrics.json"))

    # Classification report
    report = classification_report(y_test, y_pred, digits=4)
    save_text(report, os.path.join(args.reports_dir, "classification_report.txt"))

    # Confusion matrix
    labels_sorted = sorted(df_test[TARGET_COL].unique().tolist())
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    plot_confusion(cm, labels_sorted, os.path.join(args.reports_dir, "confusion_matrix.png"))

    # SHAP no teste (amostra para velocidade)
    sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_path = explain_with_shap(
        pipeline, sample, feature_names=X_test.columns.tolist(), out_dir=args.reports_dir
    )

    # Saída no console
    logging.info("==== Resultados finais (TEST) ====")
    for k, v in metrics.items():
        logging.info(f"{k}: {v:.4f}")
    logging.info("Relatórios salvos em '%s'", args.reports_dir)
    if shap_path:
        logging.info(f"SHAP: {shap_path}")


if __name__ == "__main__":
    main()
