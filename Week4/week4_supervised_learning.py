import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_curve
)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())


# --------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# --------------------------------------------------

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


# --------------------------------------------------
# 3. DEFINE NUMERICAL AND CATEGORICAL FEATURES
# --------------------------------------------------

numerical_features = [
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


# --------------------------------------------------
# 4. PREPROCESSING
# --------------------------------------------------

numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features)
    ]
)


# --------------------------------------------------
# 5. CREATE LOGISTIC REGRESSION MODEL
# --------------------------------------------------

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)


# --------------------------------------------------
# 6. TRAIN-TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------------------------
# 7. TRAIN MODEL
# --------------------------------------------------

model.fit(X_train, y_train)

print("\nModel training completed.")


# --------------------------------------------------
# 8. MAKE PREDICTIONS
# --------------------------------------------------

y_pred = model.predict(X_test)

y_probability = model.predict_proba(X_test)[:, 1]


# --------------------------------------------------
# 9. MODEL EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred)

recall = recall_score(y_test, y_pred)

f1 = f1_score(y_test, y_pred)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)


print("\n----- MODEL PERFORMANCE -----")

print("Accuracy :", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall   :", round(recall, 4))
print("F1 Score :", round(f1, 4))
print("ROC-AUC  :", round(roc_auc, 4))


print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------------------------
# 10. CROSS VALIDATION
# --------------------------------------------------

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="accuracy"
)

print("\n----- 5-FOLD CROSS VALIDATION -----")

print("Fold Scores:")

for i, score in enumerate(cv_scores, start=1):
    print(
        f"Fold {i}: {score:.4f}"
    )

print(
    "Mean CV Accuracy:",
    round(cv_scores.mean(), 4)
)

print(
    "Standard Deviation:",
    round(cv_scores.std(), 4)
)


# --------------------------------------------------
# 11. CREATE OUTPUT FOLDER
# --------------------------------------------------

import os

output_folder = "Week4/week4_outputs"

os.makedirs(
    output_folder,
    exist_ok=True
)


# --------------------------------------------------
# 12. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test,
    y_pred
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title(
    "Confusion Matrix - Logistic Regression"
)

plt.tight_layout()

plt.savefig(
    f"{output_folder}/confusion_matrix.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 13. ROC CURVE
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure()

plt.plot(
    fpr,
    tpr,
    label=f"ROC-AUC = {roc_auc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title(
    "ROC Curve - Logistic Regression"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    f"{output_folder}/roc_curve.png",
    dpi=300
)

plt.show()


# --------------------------------------------------
# 14. SAVE PERFORMANCE RESULTS
# --------------------------------------------------

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC",
        "Mean CV Accuracy",
        "CV Standard Deviation"
    ],

    "Value": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        cv_scores.mean(),
        cv_scores.std()
    ]
})

results.to_csv(
    f"{output_folder}/model_performance.csv",
    index=False
)


print("\nResults saved successfully.")

print("\nWeek 4 task completed.")