"""
Week 3 Task: Unsupervised Learning and Clustering Analysis
Dataset: Mall Customer Segmentation Dataset (public dataset, 200 customers)
Author: Triveni Megada
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.cluster.hierarchy import dendrogram, linkage

pd.set_option("display.max_columns", None)
sns.set_style("whitegrid")
PALETTE = "Set2"

# ---------------------------------------------------------------------------
# 1. DATA ACQUISITION
# ---------------------------------------------------------------------------
df = pd.read_csv("mall.csv")
df.rename(columns={"Genre": "Gender", "Annual Income (k$)": "AnnualIncome",
                    "Spending Score (1-100)": "SpendingScore"}, inplace=True)
print("Shape:", df.shape)
print(df.head())
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe())

# ---------------------------------------------------------------------------
# 2. INITIAL EXPLORATION
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.histplot(df["Age"], bins=20, kde=True, ax=axes[0], color="#4C72B0")
axes[0].set_title("Age Distribution")
sns.histplot(df["AnnualIncome"], bins=20, kde=True, ax=axes[1], color="#55A868")
axes[1].set_title("Annual Income Distribution (k$)")
sns.histplot(df["SpendingScore"], bins=20, kde=True, ax=axes[2], color="#C44E52")
axes[2].set_title("Spending Score Distribution (1-100)")
plt.tight_layout()
plt.savefig("fig1_feature_distributions.png", dpi=150)
plt.close()

print("\nGender split:\n", df["Gender"].value_counts())

# ---------------------------------------------------------------------------
# 3. PREPROCESSING
# ---------------------------------------------------------------------------
# Clustering features: Annual Income and Spending Score are the two variables
# most directly relevant to customer segmentation for marketing purposes.
# Age is added in an extended 3-feature model later in the report.
features_2d = ["AnnualIncome", "SpendingScore"]
X_2d = df[features_2d].copy()

scaler = StandardScaler()
X_2d_scaled = scaler.fit_transform(X_2d)
print("\nScaled feature means (should be ~0):", X_2d_scaled.mean(axis=0).round(3))
print("Scaled feature stds (should be ~1):", X_2d_scaled.std(axis=0).round(3))

# ---------------------------------------------------------------------------
# 4. DETERMINING OPTIMAL K (Elbow Method + Silhouette Score)
# ---------------------------------------------------------------------------
inertias = []
sil_scores = []
k_range = range(2, 11)
for k in k_range:
    km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels = km.fit_predict(X_2d_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_2d_scaled, labels))

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].plot(list(k_range), inertias, marker="o", color="#4C72B0")
axes[0].set_title("Elbow Method: Inertia vs. k")
axes[0].set_xlabel("Number of clusters (k)")
axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
axes[0].axvline(x=5, color="red", linestyle="--", alpha=0.6, label="Chosen k=5")
axes[0].legend()

axes[1].plot(list(k_range), sil_scores, marker="o", color="#55A868")
axes[1].set_title("Silhouette Score vs. k")
axes[1].set_xlabel("Number of clusters (k)")
axes[1].set_ylabel("Average Silhouette Score")
axes[1].axvline(x=5, color="red", linestyle="--", alpha=0.6, label="Chosen k=5")
axes[1].legend()
plt.tight_layout()
plt.savefig("fig2_elbow_silhouette.png", dpi=150)
plt.close()

print("\nInertia by k:", dict(zip(k_range, [round(i, 1) for i in inertias])))
print("Silhouette by k:", dict(zip(k_range, [round(s, 3) for s in sil_scores])))

best_k = 5  # chosen based on elbow bend + high silhouette score

# ---------------------------------------------------------------------------
# 5. K-MEANS CLUSTERING (k=5)
# ---------------------------------------------------------------------------
kmeans = KMeans(n_clusters=best_k, init="k-means++", n_init=10, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_2d_scaled)
final_silhouette = silhouette_score(X_2d_scaled, df["Cluster"])
print(f"\nFinal K-Means (k={best_k}) silhouette score: {final_silhouette:.3f}")

# Cluster visualization (in original, unscaled units for interpretability)
plt.figure(figsize=(8.5, 6.5))
centers_scaled = kmeans.cluster_centers_
centers_original = scaler.inverse_transform(centers_scaled)
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="Cluster",
                 palette=PALETTE, s=70, alpha=0.8)
plt.scatter(centers_original[:, 0], centers_original[:, 1], marker="X", s=250,
            c="black", label="Centroids", edgecolor="white", linewidth=1.5)
plt.title(f"Customer Segments via K-Means Clustering (k={best_k})")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("fig3_kmeans_clusters.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. SILHOUETTE PLOT
# ---------------------------------------------------------------------------
sample_sil_values = silhouette_samples(X_2d_scaled, df["Cluster"])
plt.figure(figsize=(8, 6))
y_lower = 10
colors = sns.color_palette(PALETTE, best_k)
for i in range(best_k):
    cluster_sil_values = sample_sil_values[df["Cluster"] == i]
    cluster_sil_values.sort()
    size_cluster_i = cluster_sil_values.shape[0]
    y_upper = y_lower + size_cluster_i
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_sil_values,
                       facecolor=colors[i], edgecolor=colors[i], alpha=0.8)
    plt.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10
plt.axvline(x=final_silhouette, color="red", linestyle="--",
            label=f"Average score = {final_silhouette:.2f}")
plt.title("Silhouette Plot for K-Means Clusters (k=5)")
plt.xlabel("Silhouette Coefficient")
plt.ylabel("Cluster")
plt.legend()
plt.tight_layout()
plt.savefig("fig4_silhouette_plot.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 7. HIERARCHICAL CLUSTERING (comparison / validation)
# ---------------------------------------------------------------------------
linked = linkage(X_2d_scaled, method="ward")
plt.figure(figsize=(10, 5.5))
dendrogram(linked, truncate_mode="lastp", p=30, show_leaf_counts=True)
plt.title("Hierarchical Clustering Dendrogram (Ward linkage)")
plt.xlabel("Customers (or cluster size)")
plt.ylabel("Euclidean Distance")
plt.axhline(y=9, color="red", linestyle="--", label="Cut at 5 clusters")
plt.legend()
plt.tight_layout()
plt.savefig("fig5_dendrogram.png", dpi=150)
plt.close()

agglo = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
df["Cluster_Hier"] = agglo.fit_predict(X_2d_scaled)
agreement = (df["Cluster"].astype(str) + "_") # placeholder, real check below

# Cross-tab to check how well the two methods agree
crosstab = pd.crosstab(df["Cluster"], df["Cluster_Hier"])
print("\nCross-tab: K-Means cluster vs Hierarchical cluster assignment\n", crosstab)

plt.figure(figsize=(8.5, 6.5))
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore", hue="Cluster_Hier",
                 palette=PALETTE, s=70, alpha=0.8)
plt.title("Customer Segments via Hierarchical (Agglomerative) Clustering")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(title="Cluster")
plt.tight_layout()
plt.savefig("fig6_hierarchical_clusters.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 8. CLUSTER PROFILING
# ---------------------------------------------------------------------------
profile = df.groupby("Cluster").agg(
    count=("CustomerID", "count"),
    avg_age=("Age", "mean"),
    avg_income=("AnnualIncome", "mean"),
    avg_spending=("SpendingScore", "mean"),
    pct_female=("Gender", lambda x: (x == "Female").mean() * 100),
).round(1)
print("\nCluster profile (K-Means, k=5):\n", profile)
profile.to_csv("cluster_profile.csv")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
sns.boxplot(x="Cluster", y="Age", data=df, hue="Cluster", palette=PALETTE, legend=False, ax=axes[0])
axes[0].set_title("Age by Cluster")
sns.boxplot(x="Cluster", y="AnnualIncome", data=df, hue="Cluster", palette=PALETTE, legend=False, ax=axes[1])
axes[1].set_title("Annual Income by Cluster")
sns.boxplot(x="Cluster", y="SpendingScore", data=df, hue="Cluster", palette=PALETTE, legend=False, ax=axes[2])
axes[2].set_title("Spending Score by Cluster")
plt.tight_layout()
plt.savefig("fig7_cluster_boxplots.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 9. EXTENDED 3-FEATURE CLUSTERING (Age + Income + Spending)
# ---------------------------------------------------------------------------
features_3d = ["Age", "AnnualIncome", "SpendingScore"]
X_3d = df[features_3d].copy()
X_3d_scaled = StandardScaler().fit_transform(X_3d)

sil_3d = []
for k in k_range:
    km3 = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
    labels3 = km3.fit_predict(X_3d_scaled)
    sil_3d.append(silhouette_score(X_3d_scaled, labels3))
print("\n3-feature model silhouette by k:", dict(zip(k_range, [round(s, 3) for s in sil_3d])))

best_k_3d = 6
kmeans_3d = KMeans(n_clusters=best_k_3d, init="k-means++", n_init=10, random_state=42)
df["Cluster_3D"] = kmeans_3d.fit_predict(X_3d_scaled)
sil_3d_final = silhouette_score(X_3d_scaled, df["Cluster_3D"])
print(f"Final 3-feature K-Means (k={best_k_3d}) silhouette score: {sil_3d_final:.3f}")

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection="3d")
colors3d = sns.color_palette(PALETTE, best_k_3d)
for i in range(best_k_3d):
    sub = df[df["Cluster_3D"] == i]
    ax.scatter(sub["Age"], sub["AnnualIncome"], sub["SpendingScore"],
               label=f"Cluster {i}", color=colors3d[i], s=40, alpha=0.8)
ax.set_xlabel("Age")
ax.set_ylabel("Annual Income (k$)")
ax.set_zlabel("Spending Score")
ax.set_title(f"3-Feature K-Means Clustering (Age, Income, Spending) k={best_k_3d}")
ax.legend()
plt.tight_layout()
plt.savefig("fig8_3d_clusters.png", dpi=150)
plt.close()

df.to_csv("mall_customers_clustered.csv", index=False)
print("\nDone. All figures, profile table, and clustered dataset written.")
