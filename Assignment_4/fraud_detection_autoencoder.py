"""
=============================================================================
Fraud Detection using PyOD AutoEncoder
=============================================================================
Author      : Arun Bhaskar gadde
Description : Detects fraudulent credit card transactions using an AutoEncoder
              (unsupervised deep learning) from the PyOD library. The model
              learns the normal pattern of transactions and flags those that
              deviate significantly (high reconstruction error) as anomalies.

Dataset     : Anonymized Credit Card Transactions (Kaggle)
              https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Requirements: pip install pyod pandas numpy scikit-learn matplotlib seaborn
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from pyod.models.auto_encoder import AutoEncoder
from torch import nn

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_PATH       = r"C:\Users\arung\Documents\University of the cumberlands\7_Advance_Artificial_Intelligence\Week_7\creditcard.csv"  # Path to Kaggle dataset
CONTAMINATION   = 0.001745                  # ~0.17% fraud rate in dataset
HIDDEN_NEURONS  = [64, 32, 32, 64]          # Encoder-decoder layer sizes
EPOCHS          = 30                        # Training epochs
BATCH_SIZE      = 256                       # Mini-batch size
DROPOUT_RATE    = 0.2                       # Regularisation dropout
RANDOM_STATE    = 42                        # Reproducibility seed
OUTPUT_DIR      = "output_figures"          # Folder for saved plots


# -----------------------------------------------------------------------------
# HELPER: ensure output directory exists
# -----------------------------------------------------------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)


# -----------------------------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------------------------
def load_data(filepath: str) -> pd.DataFrame:
    """
    Load the credit card dataset from CSV.
    Expected columns: Time, V1-V28 (PCA features), Amount, Class
    Class: 0 = legitimate, 1 = fraud
    """
    print("\n[1] Loading dataset...")
    df = pd.read_csv(filepath)
    print(f"    Shape         : {df.shape}")
    print(f"    Fraud cases   : {df['Class'].sum():,}  "
          f"({df['Class'].mean()*100:.4f}%)")
    print(f"    Legit cases   : {(df['Class'] == 0).sum():,}")
    return df


# -----------------------------------------------------------------------------
# 2. PREPROCESS
# -----------------------------------------------------------------------------
def preprocess(df: pd.DataFrame):
    """
    Scale Time and Amount; return feature matrix X and label vector y.
    V1-V28 are already PCA-transformed so only Time & Amount need scaling.
    """
    print("\n[2] Preprocessing...")

    scaler = StandardScaler()

    df = df.copy()
    df["Time_scaled"]   = scaler.fit_transform(df[["Time"]])
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    # Build feature matrix (drop original Time, Amount, and label)
    feature_cols = (
        [f"V{i}" for i in range(1, 29)]
        + ["Time_scaled", "Amount_scaled"]
    )
    X = df[feature_cols].values
    y = df["Class"].values

    print(f"    Features used : {len(feature_cols)}")
    return X, y


# -----------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# -----------------------------------------------------------------------------
def split_data(X: np.ndarray, y: np.ndarray):
    """
    Stratified 80 / 20 split preserving class imbalance ratio.
    """
    print("\n[3] Splitting data  (80% train / 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train samples : {X_train.shape[0]:,}")
    print(f"    Test  samples : {X_test.shape[0]:,}")
    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------------
# 4. BUILD & TRAIN AUTOENCODER
# -----------------------------------------------------------------------------
def train_autoencoder(X_train: np.ndarray) -> AutoEncoder:
    """
    Instantiate and fit PyOD AutoEncoder on training data.

    The AutoEncoder learns to reconstruct normal transactions. Fraudulent
    transactions produce higher reconstruction errors (anomaly scores).
    """
    print("\n[4] Building and training AutoEncoder (PyOD)...")
    print(f"    Hidden layers : {HIDDEN_NEURONS}")
    print(f"    Epochs        : {EPOCHS}")
    print(f"    Batch size    : {BATCH_SIZE}")
    print(f"    Dropout rate  : {DROPOUT_RATE}")
    print(f"    Contamination : {CONTAMINATION}")


    model = AutoEncoder(
        contamination=CONTAMINATION,                    # expected fraud rate (~0.17%)
        preprocessing=True,                             # internal z-score normalisation
        lr=1e-3,                                        # Adam learning rate
        epoch_num=EPOCHS,                               # number of training epochs
        batch_size=BATCH_SIZE,                          # mini-batch size
        optimizer_name="adam",                          # optimizer algorithm
        optimizer_params={"weight_decay": 1e-5},        # L2 regularisation weight decay
        hidden_neuron_list=HIDDEN_NEURONS,              # encoder-decoder layer sizes
        hidden_activation_name="relu",                  # activation for hidden layers
        batch_norm=True,                                # batch normalisation
        dropout_rate=DROPOUT_RATE,                      # dropout for regularisation
        random_state=RANDOM_STATE,                      # reproducibility seed
        verbose=1,                                      # print training progress
    )

    model.fit(X_train)
    print("    Training complete.")
    return model


# -----------------------------------------------------------------------------
# 5. EVALUATE
# -----------------------------------------------------------------------------
def evaluate(model: AutoEncoder, X_test: np.ndarray, y_test: np.ndarray):
    """
    Generate predictions and anomaly scores; print classification metrics.
    Returns predicted labels and raw anomaly scores.
    """
    print("\n[5] Evaluating on test set...")

    y_pred   = model.predict(X_test)           # 0 = normal, 1 = outlier
    scores   = model.decision_function(X_test)  # raw reconstruction error

    roc_auc  = roc_auc_score(y_test, scores)
    avg_prec = average_precision_score(y_test, scores)

    print("\n    -- Classification Report --")
    print(classification_report(y_test, y_pred,
                                target_names=["Legitimate", "Fraud"]))
    print(f"    ROC-AUC Score            : {roc_auc:.4f}")
    print(f"    Average Precision Score  : {avg_prec:.4f}")

    return y_pred, scores


# -----------------------------------------------------------------------------
# 6. VISUALISATIONS
# -----------------------------------------------------------------------------
def plot_confusion_matrix(y_test, y_pred):
    """Plot and save a labelled confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Predicted Legit", "Predicted Fraud"],
        yticklabels=["Actual Legit", "Actual Fraud"],
    )
    plt.title("Confusion Matrix - AutoEncoder Fraud Detection")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    print(f"    Saved: {path}")
    plt.show()


def plot_roc_curve(y_test, scores):
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_test, scores)
    auc_val = roc_auc_score(y_test, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2,
             label=f"ROC Curve (AUC = {auc_val:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - AutoEncoder Fraud Detection")
    plt.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.savefig(path, dpi=150)
    print(f"    Saved: {path}")
    plt.show()


def plot_anomaly_score_distribution(scores, y_test):
    """Plot reconstruction-error distribution for legit vs fraud."""
    df_plot = pd.DataFrame({"score": scores, "label": y_test})
    df_plot["Class"] = df_plot["label"].map({0: "Legitimate", 1: "Fraud"})

    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=df_plot, x="score", hue="Class",
        bins=80, kde=True,
        palette={"Legitimate": "steelblue", "Fraud": "crimson"},
    )
    plt.xlabel("Anomaly Score (Reconstruction Error)")
    plt.ylabel("Count")
    plt.title("Anomaly Score Distribution - Legitimate vs Fraud")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "anomaly_score_distribution.png")
    plt.savefig(path, dpi=150)
    print(f"    Saved: {path}")
    plt.show()


def plot_precision_recall(y_test, scores):
    """Plot and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, scores)
    ap = average_precision_score(y_test, scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="purple", lw=2,
             label=f"PR Curve (AP = {ap:.4f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve - AutoEncoder Fraud Detection")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "precision_recall_curve.png")
    plt.savefig(path, dpi=150)
    print(f"    Saved: {path}")
    plt.show()


# -----------------------------------------------------------------------------
# MAIN PIPELINE
# -----------------------------------------------------------------------------
def main():
    print("=" * 65)
    print("   Fraud Detection  -  PyOD AutoEncoder")
    print("=" * 65)

    # Step 1 - Load
    df = load_data(DATA_PATH)

    # Step 2 - Preprocess
    X, y = preprocess(df)

    # Step 3 - Split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 4 - Train
    model = train_autoencoder(X_train)

    # Step 5 - Evaluate
    y_pred, scores = evaluate(model, X_test, y_test)

    # Step 6 - Visualise
    print("\n[6] Generating visualisations...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, scores)
    plot_anomaly_score_distribution(scores, y_test)
    plot_precision_recall(y_test, scores)

    print("\n[OK] Pipeline complete. All figures saved to:", OUTPUT_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()
