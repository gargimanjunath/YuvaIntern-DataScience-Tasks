import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
# 1. Create output folder
# --------------------------------------------------

output_dir = "Week5/week5_outputs"
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)

print("Dataset Shape:", df.shape)
print("\nMissing Values:")
print(df.isnull().sum())

# --------------------------------------------------
# 3. Select features and target
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

data = df[features + [target]].copy()

# --------------------------------------------------
# 4. Handle missing values
# --------------------------------------------------

data["Age"] = data["Age"].fillna(data["Age"].median())
data["Embarked"] = data["Embarked"].fillna(
    data["Embarked"].mode()[0]
)

# --------------------------------------------------
# 5. Encode categorical variables
# --------------------------------------------------

data = pd.get_dummies(
    data,
    columns=["Sex", "Embarked"],
    drop_first=True
)

# Convert boolean columns to integers
for column in data.columns:
    if data[column].dtype == "bool":
        data[column] = data[column].astype(int)

# --------------------------------------------------
# 6. Separate X and y
# --------------------------------------------------

X = data.drop(columns=[target])
y = data[target]

# --------------------------------------------------
# 7. Train-test split
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
# 8. Feature scaling
# --------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# --------------------------------------------------
# 9. Build Neural Network
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),

    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(1, activation="sigmoid")
])

# --------------------------------------------------
# 10. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

print("\nMODEL ARCHITECTURE")
model.summary()

# --------------------------------------------------
# 11. Early stopping
# --------------------------------------------------

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)

# --------------------------------------------------
# 12. Train model
# --------------------------------------------------

history = model.fit(
    X_train,
    y_train,
    validation_split=0.20,
    epochs=50,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)

print("\nModel training completed.")

# --------------------------------------------------
# 13. Predictions
# --------------------------------------------------

y_probability = model.predict(
    X_test,
    verbose=0
).ravel()

y_pred = (y_probability >= 0.5).astype(int)

# --------------------------------------------------
# 14. Evaluation metrics
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

print("\nMODEL PERFORMANCE")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 15. Save performance metrics
# --------------------------------------------------

performance = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

performance.to_csv(
    f"{output_dir}/model_performance.csv",
    index=False
)

# --------------------------------------------------
# 16. Confusion Matrix
# --------------------------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Survived", "Survived"]
)

disp.plot()
plt.title("Confusion Matrix - Neural Network")
plt.tight_layout()
plt.savefig(
    f"{output_dir}/confusion_matrix.png",
    dpi=300
)
plt.close()

# --------------------------------------------------
# 17. ROC Curve
# --------------------------------------------------

fpr, tpr, thresholds = roc_curve(
    y_test,
    y_probability
)

plt.figure(figsize=(7, 5))
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
plt.title("ROC Curve - Neural Network")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{output_dir}/roc_curve.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 18. Training Accuracy
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{output_dir}/training_validation_accuracy.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 19. Training Loss
# --------------------------------------------------

plt.figure(figsize=(7, 5))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.tight_layout()

plt.savefig(
    f"{output_dir}/training_validation_loss.png",
    dpi=300
)

plt.close()

# --------------------------------------------------
# 20. Save model
# --------------------------------------------------

model.save(
    f"{output_dir}/titanic_neural_network.keras"
)

print("\nOutput files saved in:", output_dir)