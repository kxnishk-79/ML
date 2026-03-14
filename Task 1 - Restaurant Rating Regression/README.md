# 🍽️ Restaurant Rating Prediction

## 📌 Project Overview
This project focuses on building a **machine learning regression model** to predict the **aggregate rating of restaurants** based on various features such as location, cost, services, votes, and cuisine types.

The goal is to analyze restaurant-related data and develop a model that can estimate ratings using structured data.

---

# 🎯 Objective
Build a machine learning model that predicts the **aggregate rating of a restaurant** based on its available features.

---

# 📊 Dataset Description
The dataset contains information about restaurants, including:

| Feature Category | Examples |
|------------------|----------|
| Location Data | City, Country Code, Latitude, Longitude |
| Pricing Data | Average Cost for Two, Price Range, Currency |
| Services | Online Delivery, Table Booking |
| Engagement | Votes |
| Cuisine | Multiple cuisine types per restaurant |

### 🎯 Target Variable
`Aggregate rating`

---

# 🧹 Data Preprocessing

Several preprocessing steps were performed to clean and prepare the dataset:

### 1️⃣ Remove Invalid Data
Restaurants with **no ratings (rating = 0)** were removed to avoid misleading training data.

### 2️⃣ Drop Irrelevant Columns
The following columns were removed because they either contained identifiers or information derived from the rating:

- Restaurant ID
- Restaurant Name
- Address
- Locality
- Locality Verbose
- Rating Color
- Rating Text
- Switch to Order Menu

### 3️⃣ Handle Missing Values
Missing values in the **Cuisines column** were filled using the most frequent cuisine.

### 4️⃣ Encode Categorical Variables
Binary categorical columns were converted into numeric format:

| Original Value | Encoded Value |
|----------------|---------------|
| Yes | 1 |
| No | 0 |

### 5️⃣ One-Hot Encoding
The following categorical columns were transformed using **One-Hot Encoding**:

- City
- Currency

---

# 🍜 Cuisine Feature Engineering

Restaurants can have **multiple cuisines** (e.g., *Italian, Pizza*).  
To properly use this information:

1. The cuisine column was split into multiple values.
2. Each cuisine type was converted into a **binary feature**.
3. Dummy variables were created for each cuisine.

This increased the feature space and allowed the model to understand the influence of different cuisines on restaurant ratings.

---

# 🤖 Machine Learning Model

### Model Used
**Random Forest Regressor**

Random Forest was selected because it:

- Handles **non-linear relationships**
- Works well with **structured datasets**
- Reduces risk of **overfitting**

---

# ⚙️ Hyperparameter Configuration

The model was tuned using the following parameters:
