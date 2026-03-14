import pandas as pd

# STEP 1 — Load Dataset

df = pd.read_csv("dataset.csv")
print("\nDataset Loaded Successfully\n")

# STEP 2 — Handle Missing Values

df['Aggregate rating'] = df['Aggregate rating'].fillna(df['Aggregate rating'].mean())
df['Average Cost for two'] = df['Average Cost for two'].fillna(
    df['Average Cost for two'].median()
)
df['Cuisines'] = df['Cuisines'].fillna("Unknown")
df['City'] = df['City'].fillna("Unknown")

# STEP 3 — Ask User Preferences

print("Enter Your Preferences\n")
preferred_cuisine = input("Cuisine: ").lower()
preferred_city = input("City: ").lower()
max_budget = int(input("Maximum Cost for Two: "))
min_rating = float(input("Minimum Rating (0–5): "))

# STEP 4 — Filter Restaurants

filtered_df = df[
    (df['Cuisines'].str.lower().str.contains(preferred_cuisine)) &
    (df['City'].str.lower().str.contains(preferred_city)) &
    (df['Average Cost for two'] <= max_budget) &
    (df['Aggregate rating'] >= min_rating)
]

# STEP 5 — Sort by Rating

filtered_df = filtered_df.sort_values(
    by='Aggregate rating',
    ascending=False
)

# STEP 6 — Display Recommendations

top_n = 5
recommendations = filtered_df.head(top_n)
print("\n" + "="*55)
print("Recommended Restaurants")
print("="*55)

if recommendations.empty:
    print("\nNo restaurants found matching your preferences.\n")

else:

    for i, (_, row) in enumerate(recommendations.iterrows(), start=1):

        print(f"\n{i}. {row['Restaurant Name']}")
        print(f"   Cuisine : {row['Cuisines']}")
        print(f"   City    : {row['City']}")
        print(f"   Rating  : {row['Aggregate rating']}")
        print(f"   Cost for Two : {row['Average Cost for two']}")

print("\n" + "="*55)