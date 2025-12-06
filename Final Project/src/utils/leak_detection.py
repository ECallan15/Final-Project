from sklearn.metrics import roc_auc_score
import pandas as pd
from typing import List, Tuple

def detect_leaks(X: pd.DataFrame, y: pd.Series, threshold: float = 0.98) -> List[Tuple[str, float]]:
    leaks = []
    for col in X.columns:
        try:
            auc = roc_auc_score(y, X[col])
            if auc > threshold:
                leaks.append((col, float(auc)))
        except Exception:
            continue
    return leaks
