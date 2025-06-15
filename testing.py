import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_curve,
    auc,
    roc_auc_score
)
from sklearn.preprocessing import label_binarize

# === Load model, scaler, and feature names ===
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("random_forest_scaler.pkl")
with open("random_forest_features.json", "r") as f:
    feature_names = json.load(f)

# === Output directory ===
os.makedirs("test_model", exist_ok=True)

# === Test files ===
test_files = [
    "labeled_1hznoise.xlsx",
    "labeled_3hznoise.xlsx",
    "labeled_5hznoise.xlsx",
]

# === Feature engineering function ===
def apply_feature_engineering(data, lags):
    new_columns = []
    for lag in lags:
        new_columns.append(data['S1'].shift(lag).rename(f'lag_S1_{lag}'))
        new_columns.append(data['S2'].shift(lag).rename(f'lag_S2_{lag}'))
        new_columns.append(data['S3'].shift(lag).rename(f'lag_S3_{lag}'))
        new_columns.append(data['w(k)'].shift(lag).rename(f'lag_w(k)_{lag}'))

    for feature in ['S1', 'S2', 'S3', 'w(k)']:
        new_columns.append(data[feature].rolling(window=500).mean().rename(f'rolling_mean_{feature}_3'))
        new_columns.append(data[feature].rolling(window=500).std().rename(f'rolling_std_{feature}_3'))
        new_columns.append(data[feature].rolling(window=1000).mean().rename(f'rolling_mean_{feature}_5'))
        new_columns.append(data[feature].rolling(window=1000).std().rename(f'rolling_std_{feature}_5'))

    data = pd.concat([data] + new_columns, axis=1)
    data = data.dropna()
    return data

accuracies = []
lag_range = list(range(1, 4))  
all_curves = []  # To store (label, fpr, mean_tpr, auc)

# === Process each test file ===
for file in test_files:
    print(f"\nProcessing {file}...")

    df = pd.read_excel(file)

    for col in ['S1', 'S2', 'S3', 'w(k)']:
        df[col] = df[col].fillna(df[col].mean())

    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").astype("Int64")
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)

    df = apply_feature_engineering(df, lag_range)

    X_test = df[feature_names]
    y_test = df["Label"]
    X_test_scaled = scaler.transform(X_test)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    accuracies.append((file, acc))
    print(f"Accuracy for {file}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # === Confusion Matrix ===
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {file}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    base_name = os.path.splitext(os.path.basename(file))[0]
    plt.savefig(f"test_model/{base_name}_confusion_matrix.png")
    plt.close()

    # === ROC-AUC Calculation ===
    classes = model.classes_
    y_test_bin = label_binarize(y_test, classes=classes)

    fpr = dict()
    tpr = dict()
    for i in range(len(classes)):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_proba[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(len(classes))]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(len(classes)):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= len(classes)

    auc_score = roc_auc_score(y_test_bin, y_proba, average='macro')
    label_map = {
    "labeled_1hznoise": "1hz",
    "labeled_3hznoise": "3hz",
    "labeled_5hznoise": "5hz"
    }
    base_label = os.path.splitext(os.path.basename(file))[0]
    label = label_map.get(base_label, base_label)
    all_curves.append((label, all_fpr, mean_tpr, auc_score))

# === Combined ROC-AUC Plot ===
plt.figure(figsize=(8, 6))
for label, fpr, tpr, auc_score in all_curves:
    plt.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {auc_score:.2f})")

plt.plot([0, 1], [0, 1], 'k--', lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
#plt.title("Macro-Average ROC Curves for All Test Files")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig("test_model/all_roc_auc_curves.png")
plt.show()

# === Accuracy Summary ===
print("\n=== Summary of Accuracies ===")
for file, acc in accuracies:
    print(f"{file}: {acc:.4f}")






"""
ArithmeticErrorimport os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import json

# === Load model, scaler, and feature names ===
model = joblib.load("random_forest_model.pkl")
scaler = joblib.load("random_forest_scaler.pkl")
with open("random_forest_features.json", "r") as f:
    feature_names = json.load(f)

# === Output directory ===
os.makedirs("test_model", exist_ok=True)

# === Test files ===
test_files = [
    "labeled_1hznoise.xlsx",
    "labeled_2hznoise.xlsx",
    "labeled_3hznoise.xlsx",
    "labeled_4hznoise.xlsx",
    "labeled_5hznoise.xlsx",
]

# === Feature engineering function ===
def apply_feature_engineering(data, lags):
    new_columns = []
    for lag in lags:
        new_columns.append(data['S1'].shift(lag).rename(f'lag_S1_{lag}'))
        new_columns.append(data['S2'].shift(lag).rename(f'lag_S2_{lag}'))
        new_columns.append(data['S3'].shift(lag).rename(f'lag_S3_{lag}'))
        new_columns.append(data['w(k)'].shift(lag).rename(f'lag_w(k)_{lag}'))
        #new_columns.append(data['Time(s)'].shift(lag).rename(f'lag_Time_{lag}'))

    for feature in ['S1', 'S2', 'S3', 'w(k)']:
        new_columns.append(data[feature].rolling(window=500).mean().rename(f'rolling_mean_{feature}_3'))
        new_columns.append(data[feature].rolling(window=500).std().rename(f'rolling_std_{feature}_3'))
        new_columns.append(data[feature].rolling(window=1000).mean().rename(f'rolling_mean_{feature}_5'))
        new_columns.append(data[feature].rolling(window=1000).std().rename(f'rolling_std_{feature}_5'))
       

    data = pd.concat([data] + new_columns, axis=1)
    data = data.dropna()
    return data

accuracies = []
lag_range = list(range(1, 4))  

# === Process each test file ===
for file in test_files:
    print(f"\nProcessing {file}...")

    df = pd.read_excel(file)

    # Clean and prepare base columns
    for col in ['S1', 'S2', 'S3', 'w(k)']:
        df[col] = df[col].fillna(df[col].mean())

    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = pd.to_numeric(df["Label"], errors="coerce").astype("Int64")
    df.dropna(subset=["Label"], inplace=True)
    df["Label"] = df["Label"].astype(int)

    # Apply feature engineering
    df = apply_feature_engineering(df, lag_range)

    # Use only the feature columns used in training
    X_test = df[feature_names]
    y_test = df["Label"]

    # Standardize
    X_test_scaled = scaler.transform(X_test)

    # Predict
    y_pred = model.predict(X_test_scaled)

    # Evaluate
    acc = accuracy_score(y_test, y_pred)
    accuracies.append((file, acc))
    print(f"Accuracy for {file}: {acc:.4f}")
    print(classification_report(y_test, y_pred))

    # Save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix: {file}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    base_name = os.path.splitext(os.path.basename(file))[0]
    plt.savefig(f"test_model/{base_name}_confusion_matrix.png")
    plt.close()

# === Summary ===
print("\n=== Summary of Accuracies ===")
for file, acc in accuracies:
    print(f"{file}: {acc:.4f}")
"""


