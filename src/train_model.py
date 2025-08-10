import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.svm import SVC
import xgboost as xgb

from data_preprocessing import load_data, preprocess

def train_and_evaluate(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Models
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "XGBoost": xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False),
        "SVM": SVC(probability=True)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

        roc_auc = roc_auc_score(y_val, y_proba) if y_proba is not None else None
        report = classification_report(y_val, y_pred)
        conf = confusion_matrix(y_val, y_pred)

        results[name] = {
            "roc_auc": roc_auc,
            "classification_report": report,
            "confusion_matrix": conf
        }
        print(f"Model: {name}")
        print(f"ROC AUC: {roc_auc}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf)
        print("-" * 40)

    return results

if __name__ == "__main__":
    df = load_data()
    X_resampled, y_resampled, selected_columns = preprocess(df)
    results = train_and_evaluate(X_resampled, y_resampled)
    # Optionally, save best model and metrics here
