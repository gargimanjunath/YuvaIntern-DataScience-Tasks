import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# SETUP
# ============================================================

OUTPUT_DIR = "Week6/week6_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

print("=" * 60)
print("TITANIC INTEGRATIVE CAPSTONE PROJECT")
print("=" * 60)


# ============================================================
# 1. DATA ACQUISITION
# ============================================================

df = pd.read_csv(DATA_URL)

print("\nDataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("EXPLORATORY DATA ANALYSIS")
print("=" * 60)

print("\nSurvival Distribution:")
print(df["Survived"].value_counts())

print("\nSurvival Rate:")
print(df["Survived"].mean())


# Age distribution
plt.figure(figsize=(8, 5))
plt.hist(df["Age"].dropna(), bins=30)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Passengers")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_distribution.png"))
plt.close()


# Survival by class
survival_class = df.groupby("Pclass")["Survived"].mean()

plt.figure(figsize=(8, 5))
survival_class.plot(kind="bar")
plt.title("Survival Rate by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "survival_by_class.png"))
plt.close()


# Correlation heatmap
numeric_cols = ["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]

plt.figure(figsize=(8, 6))
sns.heatmap(
    df[numeric_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"))
plt.close()


# ============================================================
# 3. UNSUPERVISED LEARNING - K-MEANS
# ============================================================

print("\n" + "=" * 60)
print("UNSUPERVISED LEARNING - K-MEANS")
print("=" * 60)

cluster_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

cluster_data = df[cluster_features].copy()

for col in cluster_features:
    cluster_data[col] = cluster_data[col].fillna(
        cluster_data[col].median()
    )

scaler_cluster = StandardScaler()
X_cluster = scaler_cluster.fit_transform(cluster_data)

k_values = range(2, 11)
inertia_values = []
silhouette_values = []

for k in k_values:

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(X_cluster)

    inertia_values.append(kmeans.inertia_)
    silhouette_values.append(
        silhouette_score(X_cluster, labels)
    )

best_k = list(k_values)[np.argmax(silhouette_values)]
best_silhouette = max(silhouette_values)

print("\nBest Number of Clusters:", best_k)
print("Best Silhouette Score:", round(best_silhouette, 4))


# Elbow plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), inertia_values, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "elbow_method.png"))
plt.close()


# Silhouette plot
plt.figure(figsize=(8, 5))
plt.plot(list(k_values), silhouette_values, marker="o")
plt.title("Silhouette Scores")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "silhouette_scores.png"))
plt.close()


# Final K-Means
kmeans = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_cluster)

print("\nCluster Sizes:")
print(df["Cluster"].value_counts().sort_index())


# Cluster profile
cluster_profile = df.groupby("Cluster")[
    cluster_features + ["Survived"]
].mean()

print("\nCluster Profile:")
print(cluster_profile)

cluster_profile.to_csv(
    os.path.join(OUTPUT_DIR, "cluster_profile.csv")
)


# Survival by cluster
survival_cluster = df.groupby("Cluster")["Survived"].mean()

survival_cluster.to_csv(
    os.path.join(OUTPUT_DIR, "survival_by_cluster.csv")
)

plt.figure(figsize=(8, 5))
survival_cluster.plot(kind="bar")
plt.title("Survival Rate by Cluster")
plt.xlabel("Cluster")
plt.ylabel("Survival Rate")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "survival_rate_by_cluster.png")
)
plt.close()


df.to_csv(
    os.path.join(OUTPUT_DIR, "titanic_capstone_clustered.csv"),
    index=False
)


# ============================================================
# 4. SUPERVISED LEARNING - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("SUPERVISED LEARNING - LOGISTIC REGRESSION")
print("=" * 60)

features = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

target = "Survived"

X = df[features]
y = df[target]

numeric_features = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

categorical_features = [
    "Sex",
    "Embarked"
]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "onehot",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            LogisticRegression(max_iter=1000)
        )
    ]
)

logistic_model.fit(X_train, y_train)

logistic_pred = logistic_model.predict(X_test)
logistic_prob = logistic_model.predict_proba(X_test)[:, 1]

logistic_accuracy = accuracy_score(
    y_test,
    logistic_pred
)

logistic_precision = precision_score(
    y_test,
    logistic_pred
)

logistic_recall = recall_score(
    y_test,
    logistic_pred
)

logistic_f1 = f1_score(
    y_test,
    logistic_pred
)

logistic_auc = roc_auc_score(
    y_test,
    logistic_prob
)

print("\nLOGISTIC REGRESSION PERFORMANCE")
print("Accuracy :", round(logistic_accuracy, 4))
print("Precision:", round(logistic_precision, 4))
print("Recall   :", round(logistic_recall, 4))
print("F1 Score :", round(logistic_f1, 4))
print("ROC-AUC  :", round(logistic_auc, 4))


# Logistic confusion matrix
cm = confusion_matrix(
    y_test,
    logistic_pred
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "logistic_confusion_matrix.png"
    )
)
plt.close()


# Logistic ROC curve
fpr_log, tpr_log, _ = roc_curve(
    y_test,
    logistic_prob
)

plt.figure(figsize=(8, 5))
plt.plot(
    fpr_log,
    tpr_log,
    label=f"Logistic Regression (AUC = {logistic_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "logistic_roc_curve.png"
    )
)
plt.close()


# ============================================================
# 5. DEEP LEARNING - NEURAL NETWORK
# ============================================================

print("\n" + "=" * 60)
print("DEEP LEARNING - NEURAL NETWORK")
print("=" * 60)

# Prepare data specifically for neural network
nn_df = df[
    [
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Fare",
        "Embarked",
        "Survived"
    ]
].copy()

nn_df["Age"] = nn_df["Age"].fillna(
    nn_df["Age"].median()
)

nn_df["Embarked"] = nn_df["Embarked"].fillna(
    nn_df["Embarked"].mode()[0]
)

# One-hot encoding
nn_df = pd.get_dummies(
    nn_df,
    columns=["Sex", "Embarked"],
    drop_first=True
)

X_nn = nn_df.drop("Survived", axis=1)
y_nn = nn_df["Survived"]

# Convert boolean columns to integers
X_nn = X_nn.astype(float)

X_nn_train, X_nn_test, y_nn_train, y_nn_test = train_test_split(
    X_nn,
    y_nn,
    test_size=0.20,
    random_state=42,
    stratify=y_nn
)

nn_scaler = StandardScaler()

X_nn_train = nn_scaler.fit_transform(X_nn_train)
X_nn_test = nn_scaler.transform(X_nn_test)


# Neural network architecture
nn_model = Sequential([
    Dense(
        32,
        activation="relu",
        input_shape=(X_nn_train.shape[1],)
    ),
    Dropout(0.2),

    Dense(
        16,
        activation="relu"
    ),
    Dropout(0.2),

    Dense(
        1,
        activation="sigmoid"
    )
])

nn_model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

history = nn_model.fit(
    X_nn_train,
    y_nn_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# Neural network predictions
nn_prob = nn_model.predict(
    X_nn_test,
    verbose=0
).ravel()

nn_pred = (nn_prob >= 0.5).astype(int)

nn_accuracy = accuracy_score(
    y_nn_test,
    nn_pred
)

nn_precision = precision_score(
    y_nn_test,
    nn_pred
)

nn_recall = recall_score(
    y_nn_test,
    nn_pred
)

nn_f1 = f1_score(
    y_nn_test,
    nn_pred
)

nn_auc = roc_auc_score(
    y_nn_test,
    nn_prob
)


print("\nNEURAL NETWORK PERFORMANCE")
print("Accuracy :", round(nn_accuracy, 4))
print("Precision:", round(nn_precision, 4))
print("Recall   :", round(nn_recall, 4))
print("F1 Score :", round(nn_f1, 4))
print("ROC-AUC  :", round(nn_auc, 4))


# Neural network confusion matrix
nn_cm = confusion_matrix(
    y_nn_test,
    nn_pred
)

plt.figure(figsize=(6, 5))
sns.heatmap(
    nn_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title("Neural Network Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "neural_network_confusion_matrix.png"
    )
)
plt.close()


# Neural network ROC curve
fpr_nn, tpr_nn, _ = roc_curve(
    y_nn_test,
    nn_prob
)

plt.figure(figsize=(8, 5))
plt.plot(
    fpr_nn,
    tpr_nn,
    label=f"Neural Network (AUC = {nn_auc:.3f})"
)
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Neural Network ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "neural_network_roc_curve.png"
    )
)
plt.close()


# ============================================================
# 6. TRAINING HISTORY
# ============================================================

plt.figure(figsize=(8, 5))
plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)
plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)
plt.title("Neural Network Training and Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_validation_accuracy.png"
    )
)
plt.close()


plt.figure(figsize=(8, 5))
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.title("Neural Network Training and Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "training_validation_loss.png"
    )
)
plt.close()


# Save model
nn_model.save(
    os.path.join(
        OUTPUT_DIR,
        "titanic_neural_network.keras"
    )
)


# ============================================================
# 7. MODEL COMPARISON
# ============================================================

comparison = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Neural Network"
    ],
    "Accuracy": [
        logistic_accuracy,
        nn_accuracy
    ],
    "Precision": [
        logistic_precision,
        nn_precision
    ],
    "Recall": [
        logistic_recall,
        nn_recall
    ],
    "F1 Score": [
        logistic_f1,
        nn_f1
    ],
    "ROC-AUC": [
        logistic_auc,
        nn_auc
    ]
})

comparison = comparison.round(4)

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(comparison.to_string(index=False))

comparison.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.csv"
    ),
    index=False
)


# Comparison chart
plt.figure(figsize=(10, 6))

x = np.arange(len(comparison["Model"]))
width = 0.18

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1 Score",
    "ROC-AUC"
]

for i, metric in enumerate(metrics):
    plt.bar(
        x + (i - 2) * width,
        comparison[metric],
        width,
        label=metric
    )

plt.xticks(
    x,
    comparison["Model"]
)

plt.ylabel("Score")
plt.title("Model Performance Comparison")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "model_comparison.png"
    )
)

plt.close()


# ============================================================
# 8. FINAL CAPSTONE SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("CAPSTONE SUMMARY")
print("=" * 60)

print("\nDataset:", df.shape)

print(
    "Best K:",
    best_k
)

print(
    "Silhouette Score:",
    round(best_silhouette, 4)
)

print(
    "Logistic Regression Accuracy:",
    round(logistic_accuracy, 4)
)

print(
    "Logistic Regression ROC-AUC:",
    round(logistic_auc, 4)
)

print(
    "Neural Network Accuracy:",
    round(nn_accuracy, 4)
)

print(
    "Neural Network ROC-AUC:",
    round(nn_auc, 4)
)

print(
    "\nAll Week 6 outputs saved in:",
    OUTPUT_DIR
)

print("\nWeek 6 capstone completed successfully!")