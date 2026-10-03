import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataset (using Titanic dataset like Week 1)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Basic Statistical Summary
print("--- Summary Statistics ---")
print(df.describe())

# Set visual style
sns.set_theme(style="whitegrid")

# 3. Visualization 1: Distribution of Age
plt.figure(figsize=(8, 5))
sns.histplot(df['Age'].dropna(), kde=True, color='blue', bins=30)
plt.title("Age Distribution of Passengers")
plt.xlabel("Age")
plt.ylabel("Count")
plt.savefig("age_distribution.png")
plt.show()

# 4. Visualization 2: Survival Count by Passenger Class
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Pclass', hue='Survived', palette='Set2')
plt.title("Survival Count by Passenger Class")
plt.xlabel("Passenger Class (Pclass)")
plt.ylabel("Passenger Count")
plt.legend(title="Survived", labels=["No", "Yes"])
plt.savefig("survival_by_class.png")
plt.show()

# 5. Visualization 3: Correlation Heatmap for Numerical Features
plt.figure(figsize=(8, 6))
numeric_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()