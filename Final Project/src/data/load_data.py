import pandas as pd
from typing import Tuple, Optional

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def load_all(train_path: str="data/raw/train.csv",
             test_path: str="data/raw/test.csv",
             sample_path: str="data/raw/sample_submission.csv") -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]]:
    train = load_csv(train_path)
    test = load_csv(test_path)
    try:
        sample = load_csv(sample_path)
    except Exception:
        sample = None
    return train, test, sample
