"""
Week 4 Task: Supervised Learning Model Implementation
Problem: Binary classification - predict Titanic passenger survival
Dataset: Titanic (cleaned/preprocessed in Week 1)
Author: Triveni Megada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report, roc_curve, auc,
                              RocCurveDisplay)

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")
PALETTE = "Set2"
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. PROBLEM DEFINITION
# ---------------------------------------------------------------------------
# Binary classification: predict whether a passenger survived (1) or not (0)
# based on demographic and ticket information. This is a classic, well-posed
# supervised learning problem with a clear binary target already present in
# the cleaned Week 1 dataset.

df = pd.read_csv("titanic_model_ready.csv")
print("Shape:", df.shape)
print("\nTarget distribution:\n", df["survived"].value_counts(normalize=True).round(3))

# ---------------------------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------------------------
# 'sex' (text) is redundant with 'sex_encoded' (numeric) already engineered
# in Week 1 -> drop the text column to avoid duplicate/non-numeric input
df_model = df.drop(columns=["sex"])

# Additional feature: age bucketed into life-stage bands, since survival
# likelihood is not linear across age (infants/children prioritized) —
# a tree-based model can find this on its own, but it helps the linear
# model (logistic regression) capture the same non-linearity explicitly
def age_band(age):
    if age <= 12:
        return 0  # child
    elif age <= 18:
        return 1  # teen
    elif age <= 35:
        return 2  # young adult
    elif age <= 60:
        return 3  # adult
    else:
        return 4  # senior

df_model["age_band"] = df_model["age"].apply(age_band)

# Fare per family member: a single fare often covered a whole family ticket,
# so raw fare can overstate an individual's spending power for large families
df_model["fare_per_person"] = df_model["fare"] / df_model["family_size"]

print("\nEngineered feature preview:\n",
      df_model[["age", "age_band", "fare", "family_size", "fare_per_person"]].head())

# ---------------------------------------------------------------------------
# 3. TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
feature_cols = ["pclass", "age", "sibsp", "parch", "fare", "sex_encoded",
                 "embarked_c", "embarked_q", "embarked_s", "family_size",
                 "is_alone", "age_band", "fare_per_person"]
X = df_model[feature_cols].astype(float)
y = df_model["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print("Train survival rate:", y_train.mean().round(3), "| Test survival rate:", y_test.mean().round(3))

# Scale features (needed for logistic regression; harmless for tree models)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. MODEL 1: LOGISTIC REGRESSION (baseline, interpretable)
# ---------------------------------------------------------------------------
log_reg = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train_scaled, y_train)
y_pred_lr = log_reg.predict(X_test_scaled)
y_proba_lr = log_reg.predict_proba(X_test_scaled)[:, 1]

# ---------------------------------------------------------------------------
# 5. MODEL 2: RANDOM FOREST (non-linear, handles interactions)
# ---------------------------------------------------------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=3,
                             random_state=RANDOM_STATE)
rf.fit(X_train, y_train)   # tree models don't require scaling
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 6. MODEL 3: DECISION TREE (single tree, for interpretability comparison)
# ---------------------------------------------------------------------------
dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=RANDOM_STATE)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
y_proba_dt = dt.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------------------------
# 7. EVALUATION METRICS
# ---------------------------------------------------------------------------
def evaluate(name, y_true, y_pred, y_proba):
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC_AUC": auc(*roc_curve(y_true, y_proba)[:2]),
    }

results = pd.DataFrame([
    evaluate("Logistic Regression", y_test, y_pred_lr, y_proba_lr),
    evaluate("Random Forest", y_test, y_pred_rf, y_proba_rf),
    evaluate("Decision Tree", y_test, y_pred_dt, y_proba_dt),
]).set_index("Model").round(3)
print("\nModel comparison on held-out test set:\n", results)
results.to_csv("model_comparison.csv")

# ---------------------------------------------------------------------------
# 8. CROSS-VALIDATION (5-fold, stratified)
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

cv_scores_lr = cross_val_score(LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
                                StandardScaler().fit_transform(X), y, cv=cv, scoring="accuracy")
cv_scores_rf = cross_val_score(RandomForestClassifier(n_estimators=300, max_depth=6,
                                min_samples_leaf=3, random_state=RANDOM_STATE),
                                X, y, cv=cv, scoring="accuracy")
cv_scores_dt = cross_val_score(DecisionTreeClassifier(max_depth=4, min_samples_leaf=5,
                                random_state=RANDOM_STATE), X, y, cv=cv, scoring="accuracy")

print(f"\n5-fold CV accuracy - Logistic Regression: {cv_scores_lr.mean():.3f} (+/- {cv_scores_lr.std():.3f})")
print(f"5-fold CV accuracy - Random Forest:       {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std():.3f})")
print(f"5-fold CV accuracy - Decision Tree:       {cv_scores_dt.mean():.3f} (+/- {cv_scores_dt.std():.3f})")

cv_summary = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Decision Tree"],
    "CV_Mean_Accuracy": [cv_scores_lr.mean(), cv_scores_rf.mean(), cv_scores_dt.mean()],
    "CV_Std": [cv_scores_lr.std(), cv_scores_rf.std(), cv_scores_dt.std()],
}).round(3)
cv_summary.to_csv("cv_summary.csv", index=False)

# ---------------------------------------------------------------------------
# 9. HYPERPARAMETER TUNING (Random Forest, GridSearchCV)
# ---------------------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [4, 6, 8],
    "min_samples_leaf": [1, 3, 5],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=RANDOM_STATE), param_grid,
                            cv=5, scoring="accuracy", n_jobs=-1)
grid_search.fit(X_train, y_train)
print("\nBest RF params:", grid_search.best_params_)
print("Best RF CV accuracy:", round(grid_search.best_score_, 3))

best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
y_proba_best_rf = best_rf.predict_proba(X_test)[:, 1]
tuned_result = evaluate("Tuned Random Forest", y_test, y_pred_best_rf, y_proba_best_rf)
print("\nTuned RF test performance:", tuned_result)

# ---------------------------------------------------------------------------
# 10. CONFUSION MATRICES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, y_pred) in zip(axes, [("Logistic Regression", y_pred_lr),
                                       ("Random Forest", y_pred_rf),
                                       ("Decision Tree", y_pred_dt)]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                xticklabels=["Did not survive", "Survived"],
                yticklabels=["Did not survive", "Survived"])
    ax.set_title(f"{name}\nConfusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("fig1_confusion_matrices.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 11. ROC CURVES
# ---------------------------------------------------------------------------
plt.figure(figsize=(7.5, 6.5))
for name, y_proba in [("Logistic Regression", y_proba_lr),
                        ("Random Forest", y_proba_rf),
                        ("Decision Tree", y_proba_dt)]:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random guess (AUC = 0.50)")
plt.title("ROC Curves — Model Comparison")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("fig2_roc_curves.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 12. FEATURE IMPORTANCE (Random Forest)
# ---------------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print("\nRandom Forest feature importances:\n", importances.round(3))

plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
            palette=PALETTE, legend=False)
plt.title("Random Forest Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("fig3_feature_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 13. LOGISTIC REGRESSION COEFFICIENTS (interpretability)
# ---------------------------------------------------------------------------
coefs = pd.Series(log_reg.coef_[0], index=feature_cols).sort_values()
plt.figure(figsize=(8, 6))
colors = ["#C44E52" if v < 0 else "#55A868" for v in coefs.values]
plt.barh(coefs.index, coefs.values, color=colors)
plt.title("Logistic Regression Coefficients (standardized features)")
plt.xlabel("Coefficient (positive = increases survival odds)")
plt.axvline(x=0, color="black", linewidth=0.8)
plt.tight_layout()
plt.savefig("fig4_logreg_coefficients.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 14. CROSS-VALIDATION SCORE DISTRIBUTION
# ---------------------------------------------------------------------------
cv_df = pd.DataFrame({
    "Logistic Regression": cv_scores_lr,
    "Random Forest": cv_scores_rf,
    "Decision Tree": cv_scores_dt,
})
plt.figure(figsize=(8, 5.5))
sns.boxplot(data=cv_df, palette=PALETTE)
sns.stripplot(data=cv_df, color="black", alpha=0.6, size=6)
plt.title("5-Fold Cross-Validation Accuracy Distribution")
plt.ylabel("Accuracy")
plt.tight_layout()
plt.savefig("fig5_cv_distribution.png", dpi=150)
plt.close()

print("\nDone. All figures and result tables written.")
print("\nClassification report (Random Forest, held-out test set):\n",
      classification_report(y_test, y_pred_rf, target_names=["Did not survive", "Survived"]))
