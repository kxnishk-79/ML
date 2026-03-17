# ==============================
# 1. Import Libraries
# ==============================
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report


# ==============================
# 2. Load Dataset
# ==============================
df = pd.read_csv("Dataset.csv")

print("\n========== DATASET LOADED ==========")


# ==============================
# 3. Data Preprocessing
# ==============================
df = df.dropna()

df = df[['Restaurant Name', 'City', 'Average Cost for two',
         'Cuisines', 'Aggregate rating', 'Votes', 'Price range']]

df.columns = ['restaurant_name', 'location', 'cost',
              'cuisines', 'rating', 'votes', 'price_range']

df['cuisine'] = df['cuisines'].apply(lambda x: x.split(',')[0])


# ==============================
# 4. Remove Rare Cuisines
# ==============================
cuisine_counts = df['cuisine'].value_counts()
valid_cuisines = cuisine_counts[cuisine_counts >= 100].index
df = df[df['cuisine'].isin(valid_cuisines)]

print(f"\nRemaining cuisines: {len(valid_cuisines)}")


# ==============================
# 5. Features & Target
# ==============================
X = df[['restaurant_name', 'location', 'cost', 'rating', 'votes', 'price_range']]
y = df['cuisine']


# ==============================
# 6. Train-Test Split
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ==============================
# 7. Preprocessing Pipeline
# ==============================
preprocessor = ColumnTransformer(
    transformers=[
        ('name_tfidf', TfidfVectorizer(max_features=500, ngram_range=(1,2)), 'restaurant_name'),
        ('location_ohe', OneHotEncoder(handle_unknown='ignore'), ['location']),
        ('num', StandardScaler(), ['cost', 'rating', 'votes', 'price_range'])
    ]
)


# ==============================
# 8. Model Pipeline
# ==============================
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(max_iter=2000))
])


# ==============================
# 9. Train Model
# ==============================
print("\n========== TRAINING MODEL ==========")
model.fit(X_train, y_train)


# ==============================
# 10. Predictions
# ==============================
y_pred = model.predict(X_test)


# ==============================
# 11. Evaluation
# ==============================
print("\n========== MODEL PERFORMANCE ==========\n")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy  : {accuracy:.2f}")
print(f"Precision : {precision:.2f}")
print(f"Recall    : {recall:.2f}")

print("\n---------- Classification Report ----------\n")
print(classification_report(y_test, y_pred))


# ==============================
# 12. Sample Prediction
# ==============================
sample = X_test.iloc[0:1]
prediction = model.predict(sample)

print("\n========== SAMPLE PREDICTION ==========")
print(f"Predicted Cuisine: {prediction[0]}")


# ==============================
# 13. Final Message
# ==============================
print("\n========== PROCESS COMPLETED ==========")