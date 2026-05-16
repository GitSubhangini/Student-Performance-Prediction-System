# ============================================================
#  test.py
#  PURPOSE: Quick sanity check — load dataset and print first rows
#  RUN:     python test.py
# ============================================================

import pandas as pd   # pandas helps us load and work with CSV data

# Load the CSV file into a DataFrame (like a table in Python)
data = pd.read_csv("data/StudentsPerformance.csv")

# Show the first 5 rows of the dataset
print("=" * 60)
print("  DATASET PREVIEW (First 5 Rows)")
print("=" * 60)
print(data.head())

print("\n  Shape (rows, columns):", data.shape)
print("  Columns:", list(data.columns))
print("\n[OK] Dataset loaded successfully!")
