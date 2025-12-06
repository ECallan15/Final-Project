from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, average_precision_score)
import pandas as pd
from typing import Dict, Tuple, Optional

def model_metrics(y_true, y_pred, y_prob=None) -> Dict[str, float]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": float(tn) / (tn + fp) if (tn + fp) > 0 else 0.0,
        "f1": f1_score(y_true, y_pred, zero_division=0)}
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        metrics["avg_precision"] = average_precision_score(y_true, y_prob)
    return metrics

def metrics_table(models_dict, y_val):
    out = pd.DataFrame()
    for name, (y_pred, y_prob) in models_dict.items():
        out[name] = model_metrics(y_val, y_pred, y_prob)
    return out.T
