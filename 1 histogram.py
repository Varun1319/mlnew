import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)

df = data["data"]
df["MedHouseVal"] = data["target"]

print("Generating Histograms")
for col in df.columns:
    plt.figure(figsize=(8, 5))
    plt.hist(df[col], bins=50)
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.grid(axis="x")
    plt.show()

print("Generating Box Plots")
for col in df.columns:
    plt.figure(figsize=(8, 5))
    plt.boxplot(df[col])
    plt.xlabel(col)
    plt.show()

print("Detecting Outliers")
for col in df.columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_bound = Q3 + 1.5 * IQR
    lower_bound = Q1 - 1.5 * IQR
    print(f"{col}:")
    print(f"upper bound: {upper_bound}, lower_bound: {lower_bound}")
    print(
        f"Outliers: {len([i for i in df[col] if i < lower_bound or i > upper_bound])}"
    )
