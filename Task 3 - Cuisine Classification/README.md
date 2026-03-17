# 🍽️ Cuisine Classification Model

## 🎯 Objective

Develop a machine learning model to classify restaurants based on their cuisines.

---

## 🧠 Project Overview

This project predicts the **primary cuisine** of a restaurant using features like:

* 🏷️ Restaurant Name
* 📍 Location
* 💰 Cost
* ⭐ Rating
* 👍 Votes
* 💲 Price Range

---

## ⚙️ Workflow

### 🔹 1. Data Preprocessing

* Removed missing values
* Selected relevant columns
* Extracted primary cuisine from multiple labels

---

### 🔹 2. Feature Engineering

* **TF-IDF** for restaurant names (text processing)
* **One-Hot Encoding** for location
* **Standard Scaling** for numerical features

---

### 🔹 3. Handling Class Imbalance

* Filtered out rare cuisines
* Reduced noise and improved model learning

---

### 🔹 4. Model Building

* Used **Logistic Regression**
* Implemented using a **Pipeline**
* Combined preprocessing + model into a single workflow

---

## 📊 Model Performance

| Metric    | Score   |
| --------- | ------- |
| Accuracy  | **61%** |
| Precision | **63%** |
| Recall    | **61%** |

---

## 📈 Key Insights

* ✅ Strong performance on:

  * North Indian
  * Pizza
  * Bakery
  * Ice Cream

* ⚠️ Weak performance on:

  * Continental
  * Desserts
  * Mughlai

* 📉 Reason:

  * Class imbalance
  * Overlapping cuisine features

---

## 🚀 Improvements Made

* Added meaningful features (rating, votes, price range)
* Applied TF-IDF with n-grams
* Used structured preprocessing pipeline
* Selected a model suitable for text-based classification

---

## 🧾 Conclusion

The model effectively classifies restaurant cuisines with moderate accuracy.
Performance is influenced by data imbalance and feature limitations, but overall demonstrates a strong machine learning pipeline suitable for real-world datasets.

---

## 👨‍💻 Author

**Kanishk Bhatt**
