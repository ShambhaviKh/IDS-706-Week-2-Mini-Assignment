#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

#Import the Dataset
df = pd.read_csv('/Users/shambhavikhanna/Downloads/amazon_products_sales_data.csv')

#Inspect the Dataset

#Initial look
df.head()
print("\nData Info:\n")
df.info()
print("\nData Description:\n", df.describe())

#checking for missing values and duplicates
print("\nMissing values:\n", df.isnull().sum())
print("\nDuplicates:", df.duplicated().sum())

#Drop duplicates
df = df.drop_duplicates()

#Rename columns for better understanding
df = df.rename(columns={
    'title': 'product_name',
    'rating': 'product_ratings',
    'reviews': 'total_reviews',
    'purchased_last_month': 'last_month_picks'
})

#Clean the Data
#Convert numeric columns and remove symbols
numeric_cols = ['product_ratings','total_reviews','discounted_price',
                'original_price','discount_percentage','is_sponsored','last_month_picks']

for col in numeric_cols:
    df[col] = df[col].astype(str).str.replace('[\$,]', '', regex=True).str.strip()
    df[col] = pd.to_numeric(df[col], errors='coerce')

#Fill NaNs
df[numeric_cols] = df[numeric_cols].fillna(0)

#Fill categorical NaNs
categorical_cols = df.select_dtypes(include='object').columns
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

#Checking for invalid data

#Ratings outside this range are likely invalid entries
df = df[(df['product_ratings'] >= 0) & (df['product_ratings'] <= 5)]

#Remove rows where discounted_price is negative
df = df[df['discounted_price'] >= 0]

# Remove rows where original_price is negative
df = df[df['original_price'] >= 0]

#Converting target for ML classification
df['is_best_seller'] = df['is_best_seller'].map({True:1, False:0}).fillna(0)

#final shape check
print("\nRows after cleaning:", df.shape[0])

                  #FEATURE ENGINEERING & EXPLORATION
#Checking relations- Correlation Heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Numeric Features")
plt.show()                   
#Discount amount(how much discount given)
df['discount_amount'] = df['original_price'] - df['discounted_price']
print("\nTop 5 discount amounts:\n", df[['product_name','discount_amount']].sort_values(by='discount_amount', ascending=False).head())
#Reviews per rating (to see if more reviews correlate with higher ratings)
df['reviews_per_rating'] = df['total_reviews'] / (df['product_ratings'] + 0.1)
print("\nTop 5 reviews per rating:\n", df[['product_name','reviews_per_rating']].sort_values(by='reviews_per_rating', ascending=False).head())

                  #APPYLING FILTERS & GROUPINGS

#High-rated products- (>4.5) with more than 100 reviews
high_rated = df[(df["product_ratings"] >= 4.5) & (df["total_reviews"] > 100)]
print("\nHigh-rated products:\n", high_rated[['product_name','product_ratings','total_reviews']].head())

#Average discount per category
avg_discount = df.groupby("product_category")["discount_percentage"].mean().sort_values(ascending=False)
print("\nAverage discount per category:\n", avg_discount.head())

#Average product rating per category
avg_rating_per_category = df.groupby("product_category")["product_ratings"].mean().sort_values(ascending=False)
print(avg_rating_per_category)

#Best seller count per category
bestseller_counts = df.groupby("product_category")["is_best_seller"].sum().sort_values(ascending=False)
print("\nNumber of bestsellers per category:\n", bestseller_counts.head())

#Top 10 best-selling products (by last month picks)
top_sellers = df.sort_values("last_month_picks", ascending=False).head(10)
print("\nTop 10 Best-Selling Products:\n", top_sellers[['product_name','last_month_picks']])

                  #MACHINE LEARNING ALGORITHMS
#Linear Regression
X = df[['total_reviews','discounted_price','original_price','discount_percentage','is_sponsored']]
y = df['product_ratings']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rating_model = LinearRegression()
rating_model.fit(X_train, y_train)

y_pred_lr = rating_model.predict(X_test)

print("--- Linear Regression for Ratings ---")
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print("R2 Score:", r2_score(y_test, y_pred_lr))

#Prediction vs Actual
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_lr, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Product Ratings")
plt.ylabel("Predicted Product Ratings")
plt.title("Linear Regression: Actual vs Predicted Ratings")
plt.show()

#Residual Plot
residuals_lr = y_test - y_pred_lr
plt.figure(figsize=(8,5))
plt.scatter(y_test, residuals_lr, alpha=0.5)

#Fit straight line with numpy
z = np.polyfit(y_test, residuals_lr, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--")

plt.xlabel("Actual Ratings")
plt.ylabel("Residuals")
plt.title("Residuals of Linear Regression")
plt.show()

#Decision Tree
X = df[['product_ratings','discounted_price','original_price','discount_percentage','is_sponsored']]
y = df['total_reviews']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
tree_model.fit(X_train, y_train)

y_pred_tree = tree_model.predict(X_test)

print("--- Decision Tree Regressor for Total Reviews ---")
print("MSE:", mean_squared_error(y_test, y_pred_tree))
print("R2 Score:", r2_score(y_test, y_pred_tree))

#Prediction vs Actual
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_tree, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Total Reviews")
plt.ylabel("Predicted Total Reviews")
plt.title("Decision Tree: Actual vs Predicted Reviews")
plt.show()

#Residual Plot
residuals_tree = y_test - y_pred_tree
plt.figure(figsize=(8,5))
plt.scatter(y_test, residuals_tree, alpha=0.5)

#Fit straight line with numpy
z = np.polyfit(y_test, residuals_tree, 1)
p = np.poly1d(z)
plt.plot(y_test, p(y_test), "r--")

plt.xlabel("Actual Total Reviews")
plt.ylabel("Residuals")
plt.title("Residuals of Decision Tree")
plt.show()                  
                    

                        #VISUALIZATIONS

#Distribution of product ratings (How many products have high ratings?)
plt.figure(figsize=(8,5))
sns.histplot(df['product_ratings'], bins=20, kde=True, color="red")
plt.title("Distribution of Product Ratings")
plt.xlabel("Rating")
plt.ylabel("Number of Products")
plt.show()

#Price vs Ratings (Do expensive items have better ratings?)
df['price_bin'] = pd.cut(df['discounted_price'], bins=[0,500,1000,2000,5000,10000,20000], labels=["0-500","500-1000","1000-2000","2000-5000","5000-10000","10000-20000"])
plt.figure(figsize=(10,6))
sns.boxplot(x='price_bin', y='product_ratings', data=df, palette="Set2")
# Add counts to x-axis labels
counts = df['price_bin'].value_counts().reindex(["0-500","500-1000","1000-2000","2000-5000","5000-10000","10000-20000"])
plt.xticks(range(len(counts)), [f"{cat}\n(n={counts[cat]})" for cat in counts.index])
plt.xlabel("Price Range")
plt.ylabel("Product Rating")
plt.title("Ratings Across Different Price Ranges")
plt.show()

#Category Proportion (How many categories are there and whats their proportion?)
category_counts = df['product_category'].value_counts()
plt.figure(figsize=(8,8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title('Category Proportion')
plt.show()

