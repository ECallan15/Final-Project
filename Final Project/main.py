from src.data.load_data import load_all
from src.data.preprocess import basic_clean
from src.features.build_features import add_derived_features, encode_categoricals, split_X_y
from src.models.train_and_select import split_and_scale, find_best_knn_k, train_decision_tree
from src.models.evaluate import model_metrics, metrics_table
from src.utils.leak_detection import detect_leaks
from src.visualization.plots import plot_roc_curves
import pandas as pd

def main():
    train_df, test_df, sample = load_all("data/raw/train.csv", "data/raw/test.csv", "data/raw/sample_submission.csv")
    print("Loaded data: train", train_df.shape, "test", test_df.shape)

    train_clean, test_clean, train_ids, test_ids = basic_clean(train_df, test_df)
    train_feat, test_feat = add_derived_features(train_clean, test_clean)
    train_feat, test_feat = encode_categoricals(train_feat, test_feat)

    X, y = split_X_y(train_feat, target_col="Churn")
    print("Features ready. X shape:", X.shape)

    leaks = detect_leaks(X, y, threshold=0.98)
    if leaks:
        print("Potential data leaks detected (column, auc):", leaks)
    else:
        print("No high-leak features detected.")

    X_train_s, X_val_s, y_train, y_val, scaler, numeric_cols = split_and_scale(X, y)
    X_test_s = test_feat.copy()
    if numeric_cols:
        X_test_s[numeric_cols] = scaler.transform(X_test_s[numeric_cols])
        for col in numeric_cols:
            X_test_s[col] = X_test_s[col].fillna(X_train_s[col].median())

    if "Support Calls" in X_val_s.columns:
        y_pred_baseline = (X_val_s["Support Calls"] > 3).astype(int).values
        y_prob_baseline = (X_val_s["Support Calls"] > 3).astype(float).replace({0.0:0.1, 1.0:0.9}).values
    else:
        import numpy as np
        y_pred_baseline = pd.Series(0, index=y_val.index).values
        y_prob_baseline = pd.Series(0.1, index=y_val.index).values

    knn_model, best_k = find_best_knn_k(X_train_s, y_train, X_val_s, y_val)
    y_prob_knn = knn_model.predict_proba(X_val_s)[:, 1]
    y_pred_knn = (y_prob_knn > 0.5).astype(int)

    dt = train_decision_tree(X_train_s, y_train)
    y_prob_dt = dt.predict_proba(X_val_s)[:, 1]
  
    best_thresh = 0.63
    y_pred_dt = (y_prob_dt > best_thresh).astype(int)

    models = {
        "Baseline": (y_pred_baseline, y_prob_baseline),
        f"k-NN (k={best_k})": (y_pred_knn, y_prob_knn),
        "Decision Tree": (y_pred_dt, y_prob_dt)
    }
    table = metrics_table(models, y_val)
    print(table)

    fi = pd.Series(dt.feature_importances_, index=X_train_s.columns).sort_values(ascending=False)
    print("Top features (Decision Tree):")
    print(fi.head(20))

    plot_roc_curves(y_val, {"k-NN": y_prob_knn, "Decision Tree": y_prob_dt, "Baseline": y_prob_baseline})

    test_preds = knn_model.predict(X_test_s)
    submission = pd.DataFrame({"CustomerID": test_ids, "Churn": test_preds})
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")
    print(submission['Churn'].value_counts())

if __name__ == "__main__":
    main()
