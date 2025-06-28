# Titanic Data Cleaning and EDA
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Set plot style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

# Step 1: Load the dataset
df = pd.read_csv("train.csv")

# Step 2: Initial Exploration
print("🔹 First 5 rows:")
print(df.head())
print("\n🔹 Data Info:")
print(df.info())
print("\n🔹 Missing Values:")
print(df.isnull().sum())

# Step 3: Data Cleaning
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

print("\n✅ Cleaned data preview:")
print(df.head())

# Step 4: Exploratory Data Analysis (EDA)

# 4.1 Survival Count
sns.countplot(x="Survived", data=df)
plt.title("Survival Count")
plt.xlabel("Survived (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# 4.2 Survival by Gender
sns.countplot(x="Sex", hue="Survived", data=df)
plt.title("Survival by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(["Not Survived", "Survived"])
plt.show()

# 4.3 Survival by Passenger Class
sns.countplot(x="Pclass", hue="Survived", data=df)
plt.title("Survival by Passenger Class")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.legend(["Not Survived", "Survived"])
plt.show()

# 4.4 Age Distribution
sns.histplot(df["Age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# 4.5 Age vs Survival
sns.boxplot(x="Survived", y="Age", data=df)
plt.title("Age vs Survival")
plt.xlabel("Survived")
plt.ylabel("Age")
plt.show()

# 4.6 Embarked vs Survival
sns.countplot(x="Embarked", hue="Survived", data=df)
plt.title("Survival by Embarkation Port")
plt.xlabel("Embarked")
plt.ylabel("Count")
plt.legend(["Not Survived", "Survived"])
plt.show()

# 4.7 Correlation Heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

# Step 5: Save Cleaned Data
df.to_csv("cleaned_titanic.csv", index=False)
print("\n✅ Cleaned data saved as 'cleaned_titanic.csv'.")

# Step 6: Key Insights
print("\n📌 Key Insights:")
print("1. Females had a much higher survival rate than males.")
print("2. First-class passengers were more likely to survive.")
print("3. Most third-class passengers did not survive.")
print("4. Children had a better survival chance than older adults.")
print("5. Passengers who boarded at Cherbourg (C) had better survival odds.")
print("\n📄 Preview of cleaned_titanic.csv:")
print(pd.read_csv("cleaned_titanic.csv").head())
df.to_csv("cleaned_titanic.csv", index=False)

