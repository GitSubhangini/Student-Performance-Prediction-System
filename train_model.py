# ============================================================
#  train_model.py
#  PURPOSE: Train a Linear Regression model to predict math score
#           from reading score and writing score, then save it.
#  RUN:     python train_model.py
# ============================================================

# ── Imports ────────────────────────────────────────────────
import os
import pandas as pd
import numpy as np
import joblib                               # to save/load the model

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ── 1. LOAD DATA ───────────────────────────────────────────
print("=" * 55)
print("  STUDENT PERFORMANCE — MODEL TRAINING")
print("=" * 55)

print("\n📂 Step 1: Loading dataset...")
data = pd.read_csv("data/StudentsPerformance.csv")
print(f"   Rows: {data.shape[0]}  |  Columns: {data.shape[1]}")

# ── 2. CHECK FOR MISSING VALUES ────────────────────────────
print("\n🧹 Step 2: Checking for missing values...")
print(data.isnull().sum())
# Drop rows with missing values if any exist
data.dropna(inplace=True)
print("   ✅ No missing values after cleaning.")

# ── 3. DEFINE FEATURES AND TARGET ─────────────────────────
# Features (X) = what we give the model as INPUT
# Target   (y) = what we want the model to PREDICT
#
# We use:  reading score + writing score  →  predict math score

print("\n⚙️  Step 3: Defining Features (X) and Target (y)...")
X = data[['reading score', 'writing score']]   # input features
y = data['math score']                          # what we want to predict

print(f"   Features: {list(X.columns)}")
print(f"   Target  : math score")

# ── 4. SPLIT DATA ─────────────────────────────────────────
# 80% of data → model learns from it  (training set)
# 20% of data → we test accuracy on it (testing set)
print("\n✂️  Step 4: Splitting data (80% train / 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42   # fixed seed → same split every run
)
print(f"   Training samples : {len(X_train)}")
print(f"   Testing  samples : {len(X_test)}")

# ── 5. TRAIN MODEL ────────────────────────────────────────
# Linear Regression finds the best straight line through data.
# Formula:  math_score = a*reading + b*writing + c
print("\n🤖 Step 5: Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)   # ← the model LEARNS here
print("   ✅ Model trained!")

# ── 6. EVALUATE MODEL ────────────────────────────────────
print("\n📈 Step 6: Evaluating model on test data...")
y_pred = model.predict(X_test)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# model.score() returns R² — how well the model explains the data
# 1.0 = perfect | 0.0 = no better than guessing the average
score = model.score(X_test, y_test)

print(f"\n   ┌──────────────────────────────────────┐")
print(f"   │  R² Score (Accuracy) : {score:.4f}        │")
print(f"   │  Mean Absolute Error : {mae:.4f}        │")
print(f"   │  RMSE                : {rmse:.4f}        │")
print(f"   └──────────────────────────────────────┘")
print(f"\n   ✅ Model Accuracy: {score * 100:.2f}%")

# ── 7. SAVE MODEL ─────────────────────────────────────────
# joblib saves the trained model to disk so Flask can load it later
os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/model.pkl")
print("\n💾 Step 7: Model saved → model/model.pkl")

# ── 8. QUICK TEST PREDICTION ──────────────────────────────
print("\n🔮 Step 8: Quick test prediction...")
sample = [[85, 90]]   # reading=85, writing=90
predicted = model.predict(sample)[0]
print(f"   Input  : reading=85, writing=90")
print(f"   Predicted Math Score: {predicted:.2f}")

print("\n" + "=" * 55)
print("  🎉 Training complete! Run app.py for the web app.")
print("=" * 55 + "\n")
