import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

# Load dataset
df = pd.read_csv("Dataset .csv")

# Basic information
print("Shape of dataset:", df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nFirst 5 Rows:")
print(df.head())

# Remove unrated restaurants (rating = 0)
df = df[df["Aggregate rating"] > 0]

print("\nAfter removing unrated restaurants:")
print("New shape:", df.shape)

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

print("\nAfter dropping unnecessary columns:")
print("New shape:", df.shape)

print("\nRemaining columns:")
print(df.columns)

# Fill missing values in Cuisines
df["Cuisines"] = df["Cuisines"].fillna(df["Cuisines"].mode()[0])

# Convert Yes/No columns to 1/0
df["Has Table booking"] = df["Has Table booking"].map({"Yes": 1, "No": 0})
df["Has Online delivery"] = df["Has Online delivery"].map({"Yes": 1, "No": 0})
df["Is delivering now"] = df["Is delivering now"].map({"Yes": 1, "No": 0})

# One-hot encode City and Currency
df = pd.get_dummies(df, columns=["City", "Currency"], drop_first=True)

print("\nAfter Encoding:")
print("Shape:", df.shape)
print("\nData types:")
print(df.dtypes)

df = df.drop("Cuisines", axis=1)

print(df.select_dtypes(include="object").columns)

# Define X and y
X = df.drop("Aggregate rating", axis=1)
y = df["Aggregate rating"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train improved model
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42
)

model.fit(X_train, y_train)

print("Improved model training completed.")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate performance
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# Feature Importance
importances = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Important Features:")
print(importance_df.head(10))