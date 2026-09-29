import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # so it saves files even with no display attached
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score,
    precision_score, recall_score, f1_score
)
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------------------
print("Step 1: Loading CIC-MalMem-2022 dataset...")

DATA_URL = "https://huggingface.co/datasets/bvk/CIC-MalMem-2022/resolve/main/MalMem2022.csv"

try:
    df = pd.read_csv(DATA_URL)
    print(f"  Loaded directly from Hugging Face: {df.shape[0]} rows, {df.shape[1]} columns")
except Exception as e:
    print(f"  Direct download failed ({e}).")
    print("  Manual fallback: download 'MalMem2022.csv' from either")
    print("    https://www.kaggle.com/datasets/luccagodoy/obfuscated-malware-memory-2022-cic")
    print("    https://www.unb.ca/cic/datasets/malmem-2022.html")
    print("  and place it in this same folder, then re-run.")
    df = pd.read_csv("MalMem2022.csv")

print("\nColumns found:", list(df.columns))
print("\nFirst few rows:\n", df.head())


print("\nStep 2: Identifying label column...")

label_col = None
for candidate in df.columns:
    if candidate.strip().lower() in ("class", "category", "label"):
        label_col = candidate
        break

if label_col is None:
    raise ValueError(
        "Could not auto-detect the label column. Look at the printed "
        "column list above and set label_col manually, e.g.:\n"
        "    label_col = 'Category'"
    )

print(f"  Using '{label_col}' as the label column.")
print(f"  Unique values sample: {df[label_col].unique()[:10]}")

# Convert to a clean binary label: Benign = 0, everything else = 1 (malicious)
df["binary_label"] = df[label_col].apply(
    lambda x: 0 if str(x).strip().lower().startswith("benign") else 1
)
print("\nClass balance:")
print(df["binary_label"].value_counts())

# ---------------------------------------------------------------------------
# 3. PREPARE FEATURES
# ---------------------------------------------------------------------------
print("\nStep 3: Preparing features...")

drop_cols = [label_col, "binary_label"]
# Drop any other obviously non-numeric identifier columns if present
for maybe_id_col in ["Category", "Class"]:
    if maybe_id_col in df.columns and maybe_id_col not in drop_cols:
        drop_cols.append(maybe_id_col)

X = df.drop(columns=drop_cols, errors="ignore")
X = X.select_dtypes(include=[np.number])  # keep numeric features only
y = df["binary_label"]

print(f"  Feature matrix: {X.shape[0]} rows, {X.shape[1]} numeric features")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# ---------------------------------------------------------------------------
# 4. TRAIN BASELINE MODEL
# ---------------------------------------------------------------------------
print("\nStep 4: Training XGBoost baseline classifier...")

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)
model.fit(X_train, y_train)
print("  Training complete.")

# ---------------------------------------------------------------------------
# 5. EVALUATE
# ---------------------------------------------------------------------------
print("\nStep 5: Evaluating...")

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

print(f"\n{'='*50}")
print("BASELINE MODEL RESULTS (report these numbers in your demo)")
print(f"{'='*50}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
print(f"\nFull classification report:\n{classification_report(y_test, y_pred)}")

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Benign", "Malicious"],
            yticklabels=["Benign", "Malicious"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Argus Baseline — Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("\nSaved confusion_matrix.png")

# ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"XGBoost (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Argus Baseline — ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
print("Saved roc_curve.png")


print("\nStep 6: Generating SHAP explainability plot...")
try:
    import shap
    explainer = shap.TreeExplainer(model)
    # Use a small sample for speed under time pressure
    sample = X_test.sample(min(200, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(sample)

    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150)
    print("  Saved shap_summary.png")
except Exception as e:
    print(f"  SHAP step skipped ({e}) — not critical, the metrics above are the core result.")

print(f"\n{'='*50}")
print("DONE. You now have a real, working baseline detector with")
print("metrics and plots ready to drop into your demo slides.")
print(f"{'='*50}")
