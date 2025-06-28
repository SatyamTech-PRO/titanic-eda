import pandas as pd

df = pd.read_csv("train.csv")
df["Age"].fillna(df["Age"].median(), inplace=True)
df["Embarked"].fillna(df["Embarked"].mode()[0], inplace=True)
df.drop("Cabin", axis=1, inplace=True)
df.drop(["PassengerId", "Name", "Ticket"], axis=1, inplace=True)

df.to_csv("cleaned_titanic.csv", index=False)
print("✅ cleaned_titanic.csv created!")
