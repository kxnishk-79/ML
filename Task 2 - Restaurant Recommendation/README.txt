Restaurant Recommendation System

Project Overview
This project implements a simple restaurant recommendation system using Python and the Pandas library. The system recommends restaurants to users based on their preferences such as cuisine type, city, budget, and minimum rating.

The program reads a dataset of restaurants and filters the data based on the user's input. It then ranks the restaurants according to their ratings and displays the top recommendations.

Objective
Create a restaurant recommendation system based on user preferences.

Features
- Loads and processes a restaurant dataset
- Handles missing values in the dataset
- Accepts user preferences as input
- Filters restaurants based on cuisine, city, budget, and rating
- Ranks restaurants based on their aggregate rating
- Displays the top recommended restaurants in a clean format

Dataset Information
The dataset contains information about restaurants such as:
- Restaurant Name
- City
- Cuisines
- Average Cost for Two
- Aggregate Rating
- Votes

Libraries Used
- Python
- Pandas

Program Workflow
1. Load the dataset using Pandas
2. Handle missing values in important columns
3. Ask the user for preferences:
   - Preferred cuisine
   - Preferred city
   - Maximum budget
   - Minimum rating
4. Filter restaurants that match the user preferences
5. Sort the filtered restaurants by rating
6. Display the top recommended restaurants

Example Input
Cuisine: italian
City: delhi
Maximum Cost for Two: 2000
Minimum Rating: 4

Example Output
Recommended Restaurants

1. Big Chill
   Cuisine : Italian, Continental
   City    : New Delhi
   Rating  : 4.6
   Cost for Two : 1500

2. Spezia Bistro
   Cuisine : Cafe, Continental, Chinese, Italian
   City    : New Delhi
   Rating  : 4.6
   Cost for Two : 900

Evaluation
The system evaluates recommendations based on how well they match the user's preferences. Restaurants that satisfy the cuisine, city, budget, and rating criteria are recommended.

Conclusion
This project demonstrates a basic content-based recommendation system using filtering and ranking techniques. It is suitable as a beginner-level machine learning or data analysis project.

Future Improvements
- Add similarity-based recommendations using cosine similarity
- Build a graphical user interface
- Deploy the system as a web application