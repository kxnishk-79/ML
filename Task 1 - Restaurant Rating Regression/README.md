# 🍽️ Restaurant Rating Prediction

## 📌 Objective

The objective of this project is to build a machine learning regression model that predicts the **aggregate rating** of a restaurant based on various features such as location, cost, services, votes, and cuisine types.

---

## 📊 Dataset Overview

The dataset contains restaurant-level information including:

- Geographical details (Latitude, Longitude, Country Code, City)
- Pricing information (Average Cost for Two, Price Range, Currency)
- Service features (Table Booking, Online Delivery, Delivery Status)
- Cuisine types
- Customer engagement (Votes)

**Target Variable:**  
`Aggregate rating`

---

## 🧹 Data Preprocessing Steps

The following preprocessing steps were performed:

1. **Removed unrated restaurants** (Aggregate rating = 0).
2. Dropped irrelevant and leakage columns:
   - Restaurant ID
   - Restaurant Name
   - Address details
   - Rating text and color
3. Handled missing values in the `Cuisines` column.
4. Converted binary categorical variables (`Yes/No`) into numeric format (1/0).
5. Applied one-hot encoding to:
   - City
   - Currency
6. Performed feature engineering on the `Cuisines` column:
   - Split multiple cuisines per restaurant
   - Created dummy variables for each cuisine type

---

## 🤖 Model Selection

A **Random Forest Regressor** was selected due to:

- Its ability to capture non-linear relationships
- Strong performance on structured datasets
- Reduced risk of overfitting compared to a single decision tree

---

## ⚙️ Hyperparameter Tuning

The model was improved using the following configuration:

- `n_estimators = 300`
- `max_depth = 20`
- `min_samples_split = 5`
- `random_state = 42`

---

## 📈 Model Evaluation

The model was evaluated using:

- **Root Mean Squared Error (RMSE)**
- **R² Score**

### ✅ Final Results

- **RMSE:** 0.3240  
- **R² Score:** 0.6606  

---

## 🔎 Feature Importance Insights

Top influential features include:

- Number of Votes
- Geographical location (Longitude & Latitude)
- Average Cost for Two
- Specific cuisine types (e.g., Chinese, North Indian, Pizza, Continental)

This indicates that customer engagement, location, pricing, and cuisine type significantly influence restaurant ratings.

---

## 🏁 Conclusion

The final model explains approximately **66% of the variation** in restaurant ratings, demonstrating strong predictive performance for a regression-based internship task.

Further improvements could include:
- Advanced hyperparameter tuning
- Cross-validation
- Additional feature engineering techniques

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn