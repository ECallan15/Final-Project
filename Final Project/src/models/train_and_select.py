from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple
import numpy as np
import pandas as pd

def split_and_scale(X, y, test_size=0.3, random_state=123, stratify=True):
    numeric_cols = [c for c in X.columns if X[c].dtype in ['int64','float64']]
    strat = y if stratify else None
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size,
                                                      random_state=random_state, stratify=strat)

    scaler = StandardScaler()
    scaler.fit(X_train[numeric_cols])

    X_train_scaled = X_train.copy()
    X_val_scaled   = X_val.copy()

    X_train_scaled[numeric_cols] = scaler.transform(X_train[numeric_cols])
    X_val_scaled[numeric_cols]   = scaler.transform(X_val[numeric_cols])

    for col in numeric_cols:
        X_train_scaled[col] = X_train_scaled[col].fillna(X_train_scaled[col].median())
        X_val_scaled[col]   = X_val_scaled[col].fillna(X_train_scaled[col].median())

    return X_train_scaled, X_val_scaled, y_train, y_val, scaler, numeric_cols

def find_best_knn_k(X_train, y_train, X_val, y_val, k_values=None):
    if k_values is None:
        k_values = list(range(1, 31, 2))
    best_k = None
    best_f1 = -1
    for k in k_values:
        knn_tmp = KNeighborsClassifier(n_neighbors=k, weights='distance')
        knn_tmp.fit(X_train, y_train)
        y_pred_tmp = knn_tmp.predict(X_val)
        from sklearn.metrics import f1_score
        f1_tmp = f1_score(y_val, y_pred_tmp)
        if f1_tmp > best_f1:
            best_f1 = f1_tmp
            best_k = k
    knn = KNeighborsClassifier(n_neighbors=best_k, weights='distance')
    knn.fit(X_train, y_train)
    return knn, best_k

def train_decision_tree(X_train, y_train, max_depth=5, min_samples_leaf=100, random_state=123):
    dt = DecisionTreeClassifier(class_weight="balanced", random_state=random_state,
                                max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    dt.fit(X_train, y_train)
    return dt
