import pandas as pd
from typing import Tuple, List

DERIVED = ['Spend_per_Usage','High_Usage_High_Spend','Recency_per_Tenure','Late_Payment']

def add_derived_features(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    if all(col in train.columns for col in ['Total Spend','Usage Frequency']):
        train['Spend_per_Usage'] = train['Total Spend'] / (train['Usage Frequency'] + 1)
        test['Spend_per_Usage']  = test['Total Spend']  / (test['Usage Frequency']  + 1)

    if 'Usage Frequency' in train.columns and 'Total Spend' in train.columns:
        u_med = train['Usage Frequency'].median()
        s_med = train['Total Spend'].median()
        train['High_Usage_High_Spend'] = ((train['Usage Frequency'] > u_med) & (train['Total Spend'] > s_med)).astype(int)
        test['High_Usage_High_Spend']  = ((test['Usage Frequency'] > u_med) & (test['Total Spend'] > s_med)).astype(int)

    if 'Last Interaction' in train.columns and 'Tenure' in train.columns:
        train['Recency_per_Tenure'] = train['Last Interaction'] / (train['Tenure'] + 1)
        test['Recency_per_Tenure']  = test['Last Interaction']  / (test['Tenure'] + 1)

    if 'Payment Delay' in train.columns:
        p_med = train['Payment Delay'].median()
        train['Late_Payment'] = (train['Payment Delay'] > p_med).astype(int)
        test['Late_Payment']  = (test['Payment Delay']  > p_med).astype(int)

    for c in DERIVED:
        if c in train.columns:
            train[c] = train[c].fillna(0)
        if c in test.columns:
            test[c]  = test[c].fillna(0)

    if 'Payment Delay' in train.columns:
        train = train.drop(columns=['Payment Delay'], errors='ignore')
        test  = test.drop(columns=['Payment Delay'], errors='ignore')
    if 'Late_Payment' in train.columns:
        train = train.drop(columns=['Late_Payment'], errors='ignore')
        test  = test.drop(columns=['Late_Payment'], errors='ignore')

    return train, test

def encode_categoricals(train: pd.DataFrame, test: pd.DataFrame,
                        categorical_cols: list = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    if categorical_cols is None:
        categorical_cols = ["Gender", "Subscription Type", "Contract Length"]

    available = [c for c in categorical_cols if c in train.columns]
    if available:
        train = pd.get_dummies(train, columns=available, drop_first=True)
        test  = pd.get_dummies(test,  columns=available, drop_first=True)

    feature_cols = [c for c in train.columns if c != "Churn"]
    test = test.reindex(columns=feature_cols, fill_value=0)

    return train, test

def split_X_y(train: pd.DataFrame, target_col: str = "Churn") -> Tuple[pd.DataFrame, pd.Series]:
    X = train.drop(columns=[target_col])
    y = train[target_col].astype(int)
    return X, y
