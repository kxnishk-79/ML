Project Title: Restaurant Rating Prediction

Objective:
The objective of this project is to build a machine learning regression model that predicts the aggregate rating of a restaurant based on its features such as location, cost, services, votes, and cuisines.

Dataset Overview:
The dataset contains restaurant-level information including geographical data, pricing details, service availability, cuisine types, and user engagement (votes). The target variable for prediction is the "Aggregate rating".

Data Preprocessing Steps:
1. Removed restaurants with zero ratings to avoid training on unrated entries.
2. Dropped irrelevant and leakage columns such as restaurant ID, name, address details, and rating text.
3. Handled missing values in the Cuisines column.
4. Converted binary categorical variables (Yes/No) into numeric format (1/0).
5. Applied one-hot encoding to City and Currency columns.
6. Performed feature engineering on the Cuisines column by splitting multiple cuisines and creating dummy variables for each cuisine type.

Model Selection:
A Random Forest Regressor was used for prediction due to its ability to handle non-linear relationships and structured data effectively.

Hyperparameter Tuning:
The model was improved using:
- n_estimators = 300
- max_depth = 20
- min_samples_split = 5

Model Evaluation:
The model was evaluated using:
- Root Mean Squared Error (RMSE)
- R² Score

Final Results:
RMSE: 0.3240
R² Score: 0.6606

Interpretation:
The model explains approximately 66% of the variation in restaurant ratings. Feature importance analysis showed that the number of votes, geographical location (latitude and longitude), cost, and specific cuisine types significantly influence rating predictions.

Conclusion:
The final model demonstrates strong predictive performance for an internship-level regression task. Additional improvements could include further hyperparameter optimization and advanced feature engineering.