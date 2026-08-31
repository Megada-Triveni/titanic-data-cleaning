"""
Week 2 Task: Exploratory Data Analysis and Visualization
Dataset: Titanic Passenger Dataset (cleaned in Week 1)
Author: Triveni Megada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")
PALETTE = "Set2"

df = pd.read_csv("titanic_cleaned_full.csv")

# Reconstruct a single readable 'embarked' column from the one-hot columns
# produced during Week 1 preprocessing, for cleaner grouping/plotting in EDA
def _embarked_label(row):
    if row.get("embarked_c"):
        return "C"
    if row.get("embarked_q"):
        return "Q"
    return "S"
df["embarked"] = df.apply(_embarked_label, axis=1)

print("Shape:", df.shape)
print(df.dtypes)

# ---------------------------------------------------------------------------
# 1. BASIC STATISTICAL SUMMARY
# ---------------------------------------------------------------------------
print("\nNumeric summary:\n", df[["age", "fare_capped", "sibsp", "parch", "family_size"]].describe())

survival_rate = df["survived"].mean()
print(f"\nOverall survival rate: {survival_rate:.1%}")

print("\nSurvival rate by sex:\n", df.groupby("sex")["survived"].mean())
print("\nSurvival rate by class:\n", df.groupby("pclass")["survived"].mean())

# ---------------------------------------------------------------------------
# 2. UNIVARIATE ANALYSIS
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.histplot(df["age"], bins=30, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Distribution of Passenger Age")
axes[0].set_xlabel("Age (years)")
axes[0].set_ylabel("Number of Passengers")

sns.histplot(df["fare_capped"], bins=30, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Distribution of Fare (capped)")
axes[1].set_xlabel("Fare (GBP, outlier-capped)")
axes[1].set_ylabel("Number of Passengers")

sns.countplot(x="pclass", data=df, hue="pclass", palette=PALETTE, legend=False, ax=axes[2])
axes[2].set_title("Passenger Count by Class")
axes[2].set_xlabel("Passenger Class")
axes[2].set_ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig("fig1_univariate.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. SURVIVAL BY CATEGORICAL FEATURES
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

sns.barplot(x="sex", y="survived", data=df, hue="sex", palette=PALETTE, legend=False,
            errorbar=None, ax=axes[0])
axes[0].set_title("Survival Rate by Sex")
axes[0].set_ylabel("Survival Rate")
axes[0].set_xlabel("Sex")
axes[0].set_ylim(0, 1)
for c in axes[0].containers:
    axes[0].bar_label(c, fmt="%.0f%%", labels=[f"{v.get_height()*100:.0f}%" for v in c])

sns.barplot(x="pclass", y="survived", data=df, hue="pclass", palette=PALETTE, legend=False,
            errorbar=None, ax=axes[1])
axes[1].set_title("Survival Rate by Passenger Class")
axes[1].set_ylabel("Survival Rate")
axes[1].set_xlabel("Passenger Class")
axes[1].set_ylim(0, 1)
for c in axes[1].containers:
    axes[1].bar_label(c, fmt="%.0f%%", labels=[f"{v.get_height()*100:.0f}%" for v in c])

sns.barplot(x="embarked", y="survived", data=df, hue="embarked", palette=PALETTE, legend=False,
            errorbar=None, ax=axes[2])
axes[2].set_title("Survival Rate by Port of Embarkation")
axes[2].set_ylabel("Survival Rate")
axes[2].set_xlabel("Embarked (C=Cherbourg, Q=Queenstown, S=Southampton)")
axes[2].set_ylim(0, 1)
for c in axes[2].containers:
    axes[2].bar_label(c, fmt="%.0f%%", labels=[f"{v.get_height()*100:.0f}%" for v in c])
plt.tight_layout()
plt.savefig("fig2_survival_categorical.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 4. COMBINED EFFECT: SEX x CLASS
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(x="pclass", y="survived", hue="sex", data=df, palette=PALETTE, errorbar=None)
plt.title("Survival Rate by Passenger Class and Sex")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.legend(title="Sex")
plt.tight_layout()
plt.savefig("fig3_class_sex_interaction.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 5. AGE DISTRIBUTION BY SURVIVAL OUTCOME
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.kdeplot(data=df, x="age", hue="survived", fill=True, alpha=0.4, palette=PALETTE,
            common_norm=False)
plt.title("Age Distribution by Survival Outcome")
plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.legend(title="Survived", labels=["Survived", "Did not survive"])
plt.tight_layout()
plt.savefig("fig4_age_survival_kde.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. CORRELATION HEATMAP
# ---------------------------------------------------------------------------
numeric_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare_capped", "family_size"]
corr = df[numeric_cols].corr()
plt.figure(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True,
            cbar_kws={"label": "Pearson correlation"})
plt.title("Correlation Matrix of Numeric Features")
plt.tight_layout()
plt.savefig("fig5_correlation_heatmap.png", dpi=150)
plt.close()

print("\nCorrelation with survival:\n", corr["survived"].sort_values(ascending=False))

# ---------------------------------------------------------------------------
# 7. FARE vs AGE, colored by survival
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5.5))
sns.scatterplot(data=df, x="age", y="fare_capped", hue="survived", style="pclass",
                 palette=PALETTE, alpha=0.7)
plt.title("Fare vs Age, by Survival Outcome and Class")
plt.xlabel("Age (years)")
plt.ylabel("Fare (GBP, outlier-capped)")
plt.legend(title="Survived / Class", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig6_fare_age_scatter.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. FAMILY SIZE vs SURVIVAL
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
family_survival = df.groupby("family_size")["survived"].mean()
sns.barplot(x=family_survival.index, y=family_survival.values,
            hue=family_survival.index, palette=PALETTE, legend=False)
plt.title("Survival Rate by Family Size")
plt.xlabel("Family Size (siblings/spouses + parents/children + self)")
plt.ylabel("Survival Rate")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("fig7_family_size_survival.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. AGGREGATIONS TABLE
# ---------------------------------------------------------------------------
agg_table = df.groupby(["pclass", "sex"]).agg(
    passenger_count=("survived", "count"),
    survival_rate=("survived", "mean"),
    avg_age=("age", "mean"),
    avg_fare=("fare_capped", "mean"),
).round(2)
print("\nAggregated summary by class & sex:\n", agg_table)
agg_table.to_csv("aggregated_summary.csv")

print("\nDone. All figures and aggregation table written.")
