import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# ===============================
# 1. Load Dataset
# ===============================
df = pd.read_csv("Dataset .csv")

print("Shape of dataset:", df.shape)

# ===============================
# 2. Remove Unrated Restaurants
# ===============================
df = df[df["Aggregate rating"] > 0]

# ===============================
# 3. Drop Irrelevant Columns
# ===============================
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

print("After cleaning shape:", df.shape)

# ===============================
# 4. Handle Missing Values
# ===============================
df["Cuisines"] = df["Cuisines"].fillna(df["Cuisines"].mode()[0])

# ===============================
# 5. Encode Binary Columns
# ===============================
df["Has Table booking"] = df["Has Table booking"].map({"Yes": 1, "No": 0})
df["Has Online delivery"] = df["Has Online delivery"].map({"Yes": 1, "No": 0})
df["Is delivering now"] = df["Is delivering now"].map({"Yes": 1, "No": 0})

# ===============================
# 6. One-Hot Encode City & Currency
# ===============================
df = pd.get_dummies(df, columns=["City", "Currency"], drop_first=True)

# ===============================
# 7. PROPER CUISINE ENGINEERING
# ===============================
df["Cuisines"] = df["Cuisines"].str.split(", ")

df_exploded = df.explode("Cuisines")

cuisine_dummies = pd.get_dummies(df_exploded["Cuisines"], prefix="Cuisine")

cuisine_dummies = cuisine_dummies.groupby(df_exploded.index).max()

df = df.join(cuisine_dummies)

df = df.drop("Cuisines", axis=1)

print("After cuisine encoding shape:", df.shape)

# ===============================
# 8. Define Features and Target
# ===============================
X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

# ===============================
# 9. Train-Test Split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===============================
# 10. Improved Random Forest Model
# ===============================
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Improved model training completed.")

# ===============================
# 11. Evaluate Model
# ===============================
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ===============================
# 12. Feature Importance
# ===============================
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))