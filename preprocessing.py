import pandas as pd
import numpy as np

# 1. Data Acquisition: Load a public dataset (using Seaborn's titanic dataset or any local CSV)
# You can replace this URL with any public CSV link
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

# 2. Initial Data Exploration
print("--- Dataset Info ---")
print(df.info())

print("\n--- Missing Values Before Cleaning ---")
print(df.isnull().sum())

# 3. Data Cleaning: Handling Missing Values
# Fill missing Age values with the median age
df['Age'] = df['Age'].fillna(df['Age'].median())

# Drop rows where 'Embarked' has missing values
df = df.dropna(subset=['Embarked'])

# 4. Outlier Detection and Treatment (Using IQR method for 'Fare')
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Cap outliers instead of dropping them entirely
df['Fare'] = np.where(df['Fare'] > upper_bound, upper_bound, df['Fare'])

print("\n--- Missing Values After Cleaning ---")
print(df.isnull().sum())
print("\nPreprocessing completed successfully!")