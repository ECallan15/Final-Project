import pandas as pd
from typing import Tuple, List

NUMERIC_COLS = ["Age", "Tenure", "Usage Frequency", "Support Calls", "Payment Delay", "Total Spend", "Last Interaction"]

def basic_clean(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    train = train.copy()
    test = test.copy()

    train_ids = train["CustomerID"] if "CustomerID" in train.columns else None
    test_ids  = test["CustomerID"]  if "CustomerID"  in test.columns  else None

    train = train.drop(columns=["Customer Status_inactive", "Customer Status"], errors="ignore")
    test  = test.drop(columns=["Customer Status_inactive", "Customer Status"], errors="ignore")

    for df in (train, test):
        if "Support Calls" in df.columns:
            df["Support Calls"] = pd.to_numeric(df["Support Calls"].replace("none", 0), errors="coerce").fillna(0).astype(int)

    for df in (train, test):
        if "Payment Delay" in df.columns:
            df["Payment Delay"] = pd.to_numeric(df["Payment Delay"].replace("none", 0), errors="coerce")

    for col in NUMERIC_COLS:
        if col in train.columns:
            median = train[col].median()
            train[col] = train[col].fillna(median)
            if col in test.columns:
                test[col] = test[col].fillna(median)

    if "Last Interaction" in train.columns:
        train["Last Interaction"] = train["Last Interaction"].fillna(train["Last Interaction"].median())
        if "Last Interaction" in test.columns:
            test["Last Interaction"] = test["Last Interaction"].fillna(train["Last Interaction"].median())

    train = train.drop(columns=["CustomerID","Last Due Date","Last Payment Date"], errors="ignore")
    test  = test.drop(columns=["CustomerID","Last Due Date","Last Payment Date"], errors="ignore")

    return train, test, train_ids, test_ids
