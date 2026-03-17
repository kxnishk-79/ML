Project: Cuisine Classification Project

Objective:
Develop a machine learning model to classify restaurants based on their cuisines.

Overview:
This project builds a classification model using restaurant data such as name, location, cost, ratings, and votes to predict the primary cuisine type.

Steps Performed:

1. Data Preprocessing

   * Removed missing values
   * Selected relevant features
   * Extracted primary cuisine from multiple cuisines

2. Feature Engineering

   * Applied TF-IDF on restaurant names
   * Used One-Hot Encoding for location
   * Scaled numerical features (cost, rating, votes, price range)

3. Handling Imbalance

   * Removed cuisines with low sample count to improve model performance

4. Model Training

   * Used Logistic Regression
   * Built a pipeline for preprocessing and training

5. Evaluation

   * Accuracy: ~61%
   * Precision: ~63%
   * Recall: ~61%

Key Insights:

* Model performs well on popular cuisines like North Indian, Pizza, and Bakery
* Lower performance on rare or overlapping cuisines
* Feature engineering significantly improved performance from ~40% to ~61%

Conclusion:
The model demonstrates a solid approach to multi-class classification with real-world data challenges such as imbalance and feature limitations.

Author:
Kanishk Bhatt
