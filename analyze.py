import numpy as np
import pandas as pd

# Iris dataset URL
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

# Load dataset
df = pd.read_csv(url)

print("\nFIRST 5 ROWS:")
print(df.head())

# Convert numeric columns to NumPy array
data = df.iloc[:, 0:4].to_numpy()

print("\nDATA SHAPE:")
print(data.shape)

# Basic statistics
print("\nMEAN VALUES:")
print(np.mean(data, axis=0))

print("\nMIN VALUES:")
print(np.min(data, axis=0))

print("\nMAX VALUES:")
print(np.max(data, axis=0))

print("\nSTANDARD DEVIATION:")
print(np.std(data, axis=0))

# Correlation matrix
print("\nCORRELATION MATRIX:")
print(np.corrcoef(data.T))
