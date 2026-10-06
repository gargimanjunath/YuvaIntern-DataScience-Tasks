# ============================================================
# WEEK 3: UNSUPERVISED LEARNING AND CLUSTERING ANALYSIS
# Titanic Passenger Dataset
# ============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from scipy.cluster.hierarchy import dendrogram, linkage


# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

sns.set_theme(style="whitegrid")

output_dir = "week3_outputs"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


print("=" * 70)
print("WEEK 3 - UNSUPERVISED LEARNING AND CLUSTERING ANALYSIS")
print("Titanic Passenger Dataset")
print("=" * 70)


# ============================================================
# 1. DATA ACQUISITION
# ============================================================

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("\n" + "=" * 70)
print("1. DATA ACQUISITION")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 2. INITIAL DATA EXPLORATION
# ============================================================

print("\n" + "=" * 70)
print("2. INITIAL DATA EXPLORATION")
print("=" * 70)

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 3. DATA CLEANING AND PREPROCESSING
# ============================================================

print("\n" + "=" * 70)
print("3. DATA CLEANING AND PREPROCESSING")
print("=" * 70)


# -------------------------
# Handle missing Age values
# -------------------------

age_missing_before = df["Age"].isnull().sum()

df["Age"] = df["Age"].fillna(df["Age"].median())

age_missing_after = df["Age"].isnull().sum()

print("\nMissing Age values before:", age_missing_before)
print("Missing Age values after:", age_missing_after)


# -------------------------
# Handle missing Embarked values
# -------------------------

embarked_missing_before = df["Embarked"].isnull().sum()

df = df.dropna(subset=["Embarked"]).copy()

embarked_missing_after = df["Embarked"].isnull().sum()

print("\nMissing Embarked values before:", embarked_missing_before)
print("Missing Embarked values after:", embarked_missing_after)


# -------------------------
# Fare outlier treatment using IQR
# -------------------------

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

fare_outliers_before = (
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
).sum()

df["Fare"] = np.clip(
    df["Fare"],
    lower_bound,
    upper_bound
)

fare_outliers_after = (
    (df["Fare"] < lower_bound) |
    (df["Fare"] > upper_bound)
).sum()

print("\nFare outliers before treatment:", fare_outliers_before)
print("Fare outliers after treatment:", fare_outliers_after)

print("\nFare lower bound:", lower_bound)
print("Fare upper bound:", upper_bound)


# ============================================================
# 4. FEATURE SELECTION
# ============================================================

print("\n" + "=" * 70)
print("4. FEATURE SELECTION")
print("=" * 70)


features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

X = df[features].copy()

print("\nFeatures selected for clustering:")
print(features)

print("\nFeature data:")
print(X.head())

print("\nFeature statistics:")
print(X.describe())


# ============================================================
# 5. FEATURE SCALING
# ============================================================

print("\n" + "=" * 70)
print("5. FEATURE SCALING")
print("=" * 70)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

X_scaled_df = pd.DataFrame(
    X_scaled,
    columns=features
)

print("\nFirst 5 rows after standardization:")
print(X_scaled_df.head())

print("\nMean after scaling:")
print(X_scaled_df.mean())

print("\nStandard deviation after scaling:")
print(X_scaled_df.std())


# ============================================================
# 6. ELBOW METHOD
# ============================================================

print("\n" + "=" * 70)
print("6. ELBOW METHOD")
print("=" * 70)


inertias = []

k_values = range(2, 11)


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertias.append(kmeans.inertia_)


for k, inertia in zip(k_values, inertias):

    print(
        f"K = {k}, Inertia = {inertia:.2f}"
    )


# Create Elbow plot

plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    inertias,
    marker="o"
)

plt.title(
    "Elbow Method for Selecting Number of Clusters"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Within-Cluster Sum of Squares (Inertia)"
)

plt.xticks(list(k_values))

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "elbow_method.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 7. SILHOUETTE SCORE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("7. SILHOUETTE SCORE ANALYSIS")
print("=" * 70)


silhouette_scores = []


for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_scores.append(score)

    print(
        f"K = {k}, Silhouette Score = {score:.4f}"
    )


# Find best K

best_k = list(k_values)[
    np.argmax(silhouette_scores)
]

best_silhouette = max(
    silhouette_scores
)


print("\n" + "-" * 40)

print(
    "Best K based on Silhouette Score:",
    best_k
)

print(
    "Best Silhouette Score:",
    round(best_silhouette, 4)
)

print("-" * 40)


# Create silhouette plot

plt.figure(figsize=(9, 6))

plt.plot(
    list(k_values),
    silhouette_scores,
    marker="o"
)

plt.axvline(
    best_k,
    linestyle="--",
    label=f"Best K = {best_k}"
)

plt.title(
    "Silhouette Score for Different Numbers of Clusters"
)

plt.xlabel(
    "Number of Clusters (K)"
)

plt.ylabel(
    "Silhouette Score"
)

plt.xticks(list(k_values))

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "silhouette_scores.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 8. K-MEANS CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("8. K-MEANS CLUSTERING")
print("=" * 70)


kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

cluster_labels = kmeans.fit_predict(
    X_scaled
)


df["Cluster"] = cluster_labels


print("\nK-Means clustering completed.")


# Cluster sizes

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)


print("\nNumber of passengers in each cluster:")
print(cluster_counts)


# Cluster size plot

plt.figure(figsize=(9, 6))

cluster_counts.plot(
    kind="bar"
)

plt.title(
    "Number of Passengers in Each Cluster"
)

plt.xlabel(
    "Cluster"
)

plt.ylabel(
    "Number of Passengers"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "cluster_sizes.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 9. PCA CLUSTER VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("9. PCA CLUSTER VISUALIZATION")
print("=" * 70)


pca = PCA(
    n_components=2
)

X_pca = pca.fit_transform(
    X_scaled
)


print("\nExplained variance ratio:")
print(
    pca.explained_variance_ratio_
)


total_variance = (
    pca.explained_variance_ratio_.sum()
    * 100
)


print(
    "\nTotal variance explained by two components:",
    round(total_variance, 2),
    "%"
)


pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = (
    df["Cluster"].values
)


# PCA scatter plot

plt.figure(figsize=(10, 7))


for cluster in sorted(
    df["Cluster"].unique()
):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]

    plt.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.65
    )


plt.title(
    "Titanic Passenger Clusters Using PCA"
)

plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "kmeans_pca_clusters.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 10. CLUSTER PROFILE ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("10. CLUSTER PROFILE ANALYSIS")
print("=" * 70)


cluster_profile = (
    df.groupby("Cluster")[features]
    .mean()
    .round(2)
)


print(
    "\nAverage characteristics of each cluster:"
)

print(cluster_profile)


# Save cluster profile

cluster_profile.to_csv(
    os.path.join(
        output_dir,
        "cluster_profile.csv"
    )
)


# Cluster profile heatmap

plt.figure(figsize=(10, 6))

sns.heatmap(
    cluster_profile,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title(
    "Cluster Profile Heatmap"
)

plt.xlabel(
    "Features"
)

plt.ylabel(
    "Cluster"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "cluster_profile_heatmap.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 11. SURVIVAL ANALYSIS BY CLUSTER
# ============================================================

print("\n" + "=" * 70)
print("11. SURVIVAL ANALYSIS BY CLUSTER")
print("=" * 70)


survival_by_cluster = (
    df.groupby("Cluster")["Survived"]
    .mean()
    .mul(100)
    .round(2)
)


print("\nSurvival rate by cluster:")
print(survival_by_cluster)


survival_table = pd.DataFrame({
    "Cluster": survival_by_cluster.index,
    "Survival_Rate_Percent": survival_by_cluster.values
})


print("\nSurvival analysis table:")
print(survival_table)


# Save survival table

survival_table.to_csv(
    os.path.join(
        output_dir,
        "survival_by_cluster.csv"
    ),
    index=False
)


# Survival plot

plt.figure(figsize=(9, 6))

survival_by_cluster.plot(
    kind="bar"
)

plt.title(
    "Survival Rate by Passenger Cluster"
)

plt.xlabel(
    "Cluster"
)

plt.ylabel(
    "Survival Rate (%)"
)

plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        output_dir,
        "survival_rate_by_cluster.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 12. CLUSTER CHARACTERISTICS
# ============================================================

print("\n" + "=" * 70)
print("12. CLUSTER CHARACTERISTICS")
print("=" * 70)


for cluster in sorted(
    df["Cluster"].unique()
):

    cluster_data = cluster_profile.loc[
        cluster
    ]

    print(
        f"\nCluster {cluster}:"
    )

    print("-" * 40)

    print(
        f"Average Passenger Class: "
        f"{cluster_data['Pclass']:.2f}"
    )

    print(
        f"Average Age: "
        f"{cluster_data['Age']:.2f}"
    )

    print(
        f"Average Siblings/Spouses: "
        f"{cluster_data['SibSp']:.2f}"
    )

    print(
        f"Average Parents/Children: "
        f"{cluster_data['Parch']:.2f}"
    )

    print(
        f"Average Fare: "
        f"{cluster_data['Fare']:.2f}"
    )

    print(
        f"Survival Rate: "
        f"{survival_by_cluster.loc[cluster]:.2f}%"
    )


# ============================================================
# 13. HIERARCHICAL CLUSTERING
# ============================================================

print("\n" + "=" * 70)
print("13. HIERARCHICAL CLUSTERING")
print("=" * 70)


linked = linkage(
    X_scaled,
    method="ward"
)


plt.figure(figsize=(12, 7))


dendrogram(
    linked,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=8
)


plt.title(
    "Hierarchical Clustering Dendrogram"
)

plt.xlabel(
    "Clustered Samples"
)

plt.ylabel(
    "Euclidean Distance"
)

plt.tight_layout()


plt.savefig(
    os.path.join(
        output_dir,
        "hierarchical_dendrogram.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. SAVING RESULTS
# ============================================================

print("\n" + "=" * 70)
print("14. SAVING RESULTS")
print("=" * 70)


df.to_csv(
    os.path.join(
        output_dir,
        "titanic_week3_clustered.csv"
    ),
    index=False
)


print(
    "\nFinal clustered dataset saved."
)

print(
    "Location:",
    os.path.join(
        output_dir,
        "titanic_week3_clustered.csv"
    )
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("WEEK 3 ANALYSIS COMPLETED")
print("=" * 70)


print("\nDataset:")
print("Titanic Passenger Dataset")


print("\nNumber of observations after cleaning:")
print(len(df))


print("\nFeatures used for clustering:")
print(features)


print("\nBest number of clusters:")
print(best_k)


print("\nBest Silhouette Score:")
print(
    round(
        best_silhouette,
        4
    )
)


print("\nCluster sizes:")
print(cluster_counts)


print("\nSurvival rate by cluster:")
print(survival_by_cluster)


print("\nAll output files have been saved in:")
print(output_dir)


print("\n" + "=" * 70)
print("END OF WEEK 3 ANALYSIS")
print("=" * 70)