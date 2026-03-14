"""
Project: Restaurant Rating Prediction
Objective: Predict aggregate restaurant ratings using machine learning.
Model Used: Random Forest Regressor
Evaluation Metrics: RMSE and R² Score
"""

import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Ensure proper UTF-8 printing (for Windows consoles)
sys.stdout.reconfigure(encoding='utf-8')

# =====================================================
# 1. Load Dataset
# =====================================================
df = pd.read_csv("Dataset .csv")

print("Initial Dataset Shape:", df.shape)

# =====================================================
# 2. Data Cleaning
# =====================================================

# Remove restaurants with zero rating
df = df[df["Aggregate rating"] > 0]

# Drop irrelevant and leakage columns
df = df.drop([
    "Restaurant ID",
    "Restaurant Name",
    "Address",
    "Locality",
    "Locality Verbose",
    "Rating color",
    "Rating text",
    "Switch to order menu"
], axis=1)

print("After Cleaning Shape:", df.shape)

# =====================================================
# 3. Handle Missing Values
# =====================================================
df["Cuisines"] = df["Cuisines"].fillna(df["Cuisines"].mode()[0])

# =====================================================
# 4. Encode Categorical Variables
# =====================================================

# Convert binary Yes/No columns
df["Has Table booking"] = df["Has Table booking"].map({"Yes": 1, "No": 0})
df["Has Online delivery"] = df["Has Online delivery"].map({"Yes": 1, "No": 0})
df["Is delivering now"] = df["Is delivering now"].map({"Yes": 1, "No": 0})

# One-hot encode City and Currency
df = pd.get_dummies(df, columns=["City", "Currency"], drop_first=True)

# =====================================================
# 5. Cuisine Feature Engineering
# =====================================================

# Split multiple cuisines
df["Cuisines"] = df["Cuisines"].str.split(", ")

# Expand rows
df_exploded = df.explode("Cuisines")

# Create dummy variables
cuisine_dummies = pd.get_dummies(df_exploded["Cuisines"], prefix="Cuisine")

# Aggregate back
cuisine_dummies = cuisine_dummies.groupby(df_exploded.index).max()

# Join back to main dataframe
df = df.join(cuisine_dummies)

# Drop original Cuisines column
df = df.drop("Cuisines", axis=1)

print("After Cuisine Encoding Shape:", df.shape)

# =====================================================
# 6. Define Features and Target
# =====================================================
X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

# =====================================================
# 7. Train-Test Split
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =====================================================
# 8. Model Training (Tuned Random Forest)
# =====================================================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

# =====================================================
# 9. Model Evaluation
# =====================================================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n==============================")
print("FINAL MODEL PERFORMANCE")
print("==============================")
print(f"RMSE: {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# =====================================================
# 10. Feature Importance
# =====================================================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Influential Features:")
print(importance_df.head(10))