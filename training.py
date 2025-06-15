import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import StandardScaler, label_binarize
import json
import os
import joblib


def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)


def load_dataset(file_path):
    return pd.read_excel(file_path)


def feature_engineering(data, lags):
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


def plot_roc_curves(models, X_test, y_test, output_path="auc_roc_comparison.png"):
    # Set publication-quality parameters
    mpl.rcParams.update({
        "figure.figsize": (8, 6),
        "font.size": 14,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12
    })

    # Check for multiclass or binary classification
    if len(np.unique(y_test)) > 2:
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        is_multiclass = True
    else:
        y_test_bin = y_test
        is_multiclass = False

    plt.figure()

    for model_name, model in models.items():
        if is_multiclass:
            y_score = model.predict_proba(X_test)
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(y_score.shape[1]):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute macro average
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(y_score.shape[1])]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(y_score.shape[1]):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= y_score.shape[1]
            roc_auc_macro = auc(all_fpr, mean_tpr)
            plt.plot(all_fpr, mean_tpr, label=f"{model_name} (AUC = {roc_auc_macro:.2f})")
        else:
            y_score = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")

    # Plot styling
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.02])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for All Models")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_evaluate_models(data, lags):
    feature_columns = (
        ["S1", "S2", "S3", "w(k)"] +
        [f'rolling_mean_{f}_3' for f in ['S1', 'S2', 'S3', 'w(k)']] +
        [f'rolling_std_{f}_3' for f in ['S1', 'S2', 'S3', 'w(k)']] +
        [f'rolling_mean_{f}_5' for f in ['S1', 'S2', 'S3', 'w(k)']] +
        [f'rolling_std_{f}_5' for f in ['S1', 'S2', 'S3', 'w(k)']] +
        [f'lag_S1_{lag}' for lag in lags] +
        [f'lag_S2_{lag}' for lag in lags] +
        [f'lag_S3_{lag}' for lag in lags] +
        [f'lag_w(k)_{lag}' for lag in lags]
    )

    X = data[feature_columns]
    y = data["Label"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_names = list(X.columns)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1),
        "XGBoost": XGBClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{model_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig(f"{model_name.replace(' ', '_').lower()}_confusion_matrix.png")
        plt.close()

        # Save model and preprocessing artifacts
        base_name = model_name.replace(' ', '_').lower()
        joblib.dump(model, f"{base_name}_model.pkl")
        joblib.dump(scaler, f"{base_name}_scaler.pkl")
        with open(f"{base_name}_features.json", "w") as f:
            json.dump(feature_names, f)

        print(f"Saved {model_name}, its scaler, and feature names.")

    # Plot ROC curves for all models
    plot_roc_curves(models, X_test, y_test)
    print("Saved AUC-ROC comparison figure as 'auc_roc_comparison.png'")


def main():
    config = load_config()
    lag_range = list(range(1, 4))
    data = load_dataset(config["file_path"])
    data = feature_engineering(data, lags=lag_range)
    train_and_evaluate_models(data, lags=lag_range)


if __name__ == "__main__":
    main()
