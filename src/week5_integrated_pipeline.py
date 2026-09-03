"""
Week 5 integrated WeatherAUS project pipeline.
Reproduces the Week 4 baseline model while keeping leakage prevention explicit.
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "weatherAUS.csv"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
df.columns = [c.strip() for c in df.columns]
target = "RainTomorrow"
drop_cols = ["RainTomorrow", "RISK_MM", "Date"]
features = [c for c in df.columns if c not in drop_cols]
d = df[features + [target]].copy()
d[target] = d[target].astype(str).str.strip()
d = d[d[target].isin(["Yes", "No"])]

X, y = d[features], d[target].map({"No":0, "Yes":1})
num = X.select_dtypes(include=[np.number]).columns.tolist()
cat = X.select_dtypes(exclude=[np.number]).columns.tolist()

pre = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), num),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)
])
model = Pipeline([
    ("preprocessor", pre),
    ("model", LogisticRegression(max_iter=1000, class_weight="balanced",
                                 solver="liblinear", random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
model.fit(X_train, y_train)
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

metrics = {
    "Baseline Accuracy": accuracy_score(y_test, baseline.predict(X_test)),
    "Accuracy": accuracy_score(y_test, pred),
    "Precision": precision_score(y_test, pred, zero_division=0),
    "Recall": recall_score(y_test, pred, zero_division=0),
    "F1 Score": f1_score(y_test, pred, zero_division=0),
    "ROC-AUC": roc_auc_score(y_test, prob)
}
pd.DataFrame(list(metrics.items()), columns=["Metric","Value"]).to_csv(
    RESULTS/"integrated_model_results.csv", index=False
)
pd.DataFrame(confusion_matrix(y_test,pred),
             index=["Actual No Rain","Actual Rain"],
             columns=["Predicted No Rain","Predicted Rain"]).to_csv(
    RESULTS/"integrated_confusion_matrix.csv"
)
print(pd.Series(metrics))
