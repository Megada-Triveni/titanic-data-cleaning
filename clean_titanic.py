"""
Week 1 Task: Data Acquisition, Cleaning, and Preprocessing
Dataset: Titanic Passenger Dataset (public, via seaborn / Kaggle "Titanic - Machine
Learning from Disaster")
Author: Triveni Megada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")

# ---------------------------------------------------------------------------
# STEP 1: DATA ACQUISITION
# ---------------------------------------------------------------------------
df = sns.load_dataset("titanic")
df.to_csv("titanic_raw.csv", index=False)

print("Shape:", df.shape)
print("\nColumn dtypes:\n", df.dtypes)

# ---------------------------------------------------------------------------
# STEP 2: INITIAL DATA EXPLORATION
# ---------------------------------------------------------------------------
print("\nFirst 5 rows:\n", df.head())
print("\nSummary statistics:\n", df.describe(include="all"))

missing_summary = df.isnull().sum().sort_values(ascending=False)
missing_pct = (missing_summary / len(df) * 100).round(2)
missing_table = pd.concat([missing_summary, missing_pct], axis=1,
                           keys=["Missing Count", "Missing %"])
missing_table = missing_table[missing_table["Missing Count"] > 0]
print("\nMissing values:\n", missing_table)

# Visualize missingness
plt.figure(figsize=(9, 5))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
plt.title("Missing Value Map (raw data)")
plt.tight_layout()
plt.savefig("fig1_missing_heatmap.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
missing_table["Missing Count"].plot(kind="bar", color="#d95f02")
plt.title("Missing Values per Column")
plt.ylabel("Count of missing values")
plt.tight_layout()
plt.savefig("fig2_missing_bar.png", dpi=150)
plt.close()

# Duplicate check
n_dupes = df.duplicated().sum()
print("\nDuplicate rows:", n_dupes)

# ---------------------------------------------------------------------------
# STEP 3: HANDLING MISSING VALUES
# ---------------------------------------------------------------------------
df_clean = df.copy()

# 'deck' is ~77% missing -> not reliably imputable, drop the column
df_clean.drop(columns=["deck"], inplace=True)

# 'age' (~20% missing) -> impute with median grouped by passenger class & sex,
# since age distributions differ meaningfully across these groups
df_clean["age"] = df_clean.groupby(["pclass", "sex"])["age"].transform(
    lambda x: x.fillna(x.median())
)

# 'embarked' / 'embark_town' (2 missing) -> impute with the mode (most common port)
mode_port = df_clean["embarked"].mode()[0]
df_clean["embarked"] = df_clean["embarked"].fillna(mode_port)
df_clean["embark_town"] = df_clean["embark_town"].fillna(df_clean["embark_town"].mode()[0])

print("\nMissing values after imputation:\n", df_clean.isnull().sum())

# ---------------------------------------------------------------------------
# STEP 4: HANDLING INCONSISTENCIES / ERRONEOUS ENTRIES
# ---------------------------------------------------------------------------
# Standardize categorical text fields (case, whitespace)
for col in ["sex", "embarked", "embark_town", "class", "who", "alive"]:
    df_clean[col] = df_clean[col].astype(str).str.strip().str.lower()

# Logical consistency check: fare should not be negative or zero for paying
# passengers travelling in a cabin class; flag but do not silently alter
inconsistent_fare = df_clean[df_clean["fare"] <= 0]
print("\nRows with non-positive fare:", len(inconsistent_fare))

# Logical consistency: 'sibsp' and 'parch' should not be negative
neg_family = df_clean[(df_clean["sibsp"] < 0) | (df_clean["parch"] < 0)]
print("Rows with negative family-size counts:", len(neg_family))

# 'adult_male' should be boolean and consistent with 'age' >=18 & sex == male
mismatch = df_clean[(df_clean["adult_male"]) & (df_clean["sex"] != "male")]
print("Rows where adult_male flag conflicts with sex field:", len(mismatch))

# ---------------------------------------------------------------------------
# STEP 5: OUTLIER DETECTION AND TREATMENT
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(y=df_clean["fare"], ax=axes[0], color="#1b9e77")
axes[0].set_title("Fare - before treatment")
sns.boxplot(y=df_clean["age"], ax=axes[1], color="#7570b3")
axes[1].set_title("Age - before treatment")
plt.tight_layout()
plt.savefig("fig3_boxplots_before.png", dpi=150)
plt.close()

def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

fare_low, fare_high = iqr_bounds(df_clean["fare"])
n_fare_outliers = ((df_clean["fare"] < fare_low) | (df_clean["fare"] > fare_high)).sum()
print(f"\nFare IQR bounds: [{fare_low:.2f}, {fare_high:.2f}] -> "
      f"{n_fare_outliers} outlier rows ({n_fare_outliers/len(df_clean)*100:.1f}%)")

# Fare outliers are genuine (first-class / luxury suites really did cost far
# more) so we cap (winsorize) rather than delete, to preserve sample size
# while limiting the influence of extreme values on downstream models
df_clean["fare_capped"] = df_clean["fare"].clip(lower=fare_low, upper=fare_high)

age_low, age_high = iqr_bounds(df_clean["age"])
n_age_outliers = ((df_clean["age"] < age_low) | (df_clean["age"] > age_high)).sum()
print(f"Age IQR bounds: [{age_low:.2f}, {age_high:.2f}] -> "
      f"{n_age_outliers} outlier rows ({n_age_outliers/len(df_clean)*100:.1f}%)")
# Age outliers (elderly passengers) are plausible, real values -> keep as-is

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
sns.boxplot(y=df_clean["fare_capped"], ax=axes[0], color="#1b9e77")
axes[0].set_title("Fare - after capping (IQR winsorization)")
sns.boxplot(y=df_clean["age"], ax=axes[1], color="#7570b3")
axes[1].set_title("Age - unchanged (kept, plausible values)")
plt.tight_layout()
plt.savefig("fig4_boxplots_after.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# STEP 6: FEATURE PREPROCESSING
# ---------------------------------------------------------------------------
# Encode categorical variables
df_clean["sex_encoded"] = df_clean["sex"].map({"male": 0, "female": 1})
df_clean = pd.get_dummies(df_clean, columns=["embarked"], prefix="embarked")

# Feature engineering: family size, is_alone
df_clean["family_size"] = df_clean["sibsp"] + df_clean["parch"] + 1
df_clean["is_alone"] = (df_clean["family_size"] == 1).astype(int)

# Drop redundant / leakage-prone columns for a modeling-ready table
df_model_ready = df_clean.drop(columns=["class", "who", "alive", "embark_town",
                                         "adult_male", "alone", "fare"])
df_model_ready.rename(columns={"fare_capped": "fare"}, inplace=True)

print("\nFinal cleaned dataset shape:", df_model_ready.shape)
print("Final columns:", df_model_ready.columns.tolist())
print("\nRemaining nulls check:\n", df_model_ready.isnull().sum().sum())

df_clean.to_csv("titanic_cleaned_full.csv", index=False)
df_model_ready.to_csv("titanic_model_ready.csv", index=False)

# Distribution comparison plot (age) before vs after imputation
plt.figure(figsize=(8, 5))
sns.kdeplot(df["age"].dropna(), label="Age - raw (nulls dropped)", fill=True, alpha=0.4)
sns.kdeplot(df_clean["age"], label="Age - after group-median imputation", fill=True, alpha=0.4)
plt.title("Age Distribution: Raw vs Imputed")
plt.legend()
plt.tight_layout()
plt.savefig("fig5_age_distribution.png", dpi=150)
plt.close()

print("\nDone. Files written: titanic_cleaned_full.csv, titanic_model_ready.csv, fig1-5 PNGs")
