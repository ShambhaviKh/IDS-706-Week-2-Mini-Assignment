#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

#-------------------------------
# DATA LOADING
#-------------------------------
def load_data(file_path):
    """Load CSV file into a DataFrame"""
    df = pd.read_csv('/Users/shambhavikhanna/Downloads/amazon_products_sales_data.csv')
    return df

#-------------------------------
# DATA INSPECTION
#-------------------------------
def inspect_data(df):
    """Perform initial inspection of the dataset"""
    print("\nData Info:\n")
    df.info()
    print("\nData Description:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())

#-------------------------------
# DATA CLEANING
#-------------------------------
def clean_data(df):
    """Clean dataset: remove duplicates, rename columns, convert types, fill NaNs"""
    # Drop duplicates
    df = df.drop_duplicates()

    # Rename columns
    df = df.rename(columns={
        'title': 'product_name',
        'rating': 'product_ratings',
        'reviews': 'total_reviews',
        'purchased_last_month': 'last_month_picks'
    })

    # Convert numeric columns
    numeric_cols = ['product_ratings','total_reviews','discounted_price',
                    'original_price','discount_percentage','is_sponsored','last_month_picks']
    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace('[\$,]', '', regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill numeric NaNs
    df[numeric_cols] = df[numeric_cols].fillna(0)

    # Fill categorical NaNs
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Remove invalid entries
    df = df[(df['product_ratings'] >= 0) & (df['product_ratings'] <= 5)]
    df = df[df['discounted_price'] >= 0]
    df = df[df['original_price'] >= 0]

    # Convert target for classification
    df['is_best_seller'] = df['is_best_seller'].map({True:1, False:0}).fillna(0)

    print("\nRows after cleaning:", df.shape[0])
    return df

#-------------------------------
# FEATURE ENGINEERING
#-------------------------------
def feature_engineering(df):
    """Create new features and perform correlation analysis"""
    numeric_cols = ['product_ratings','total_reviews','discounted_price',
                    'original_price','discount_percentage','is_sponsored','last_month_picks']

    # Correlation Heatmap
    plt.figure(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()

    # Discount amount
    df['discount_amount'] = df['original_price'] - df['discounted_price']
    print("\nTop 5 discount amounts:\n", df[['product_name','discount_amount']].sort_values(by='discount_amount', ascending=False).head())

    # Reviews per rating
    df['reviews_per_rating'] = df['total_reviews'] / (df['product_ratings'] + 0.1)
    print("\nTop 5 reviews per rating:\n", df[['product_name','reviews_per_rating']].sort_values(by='reviews_per_rating', ascending=False).head())

    return df

#-------------------------------
# FILTERING & GROUPING
#-------------------------------
def filtering_grouping(df):
    """Apply filters and groupings for exploration"""
    # High-rated products
    high_rated = df[(df["product_ratings"] >= 4.5) & (df["total_reviews"] > 100)]
    print("\nHigh-rated products:\n", high_rated[['product_name','product_ratings','total_reviews']].head())

    # Average discount per category
    avg_discount = df.groupby("product_category")["discount_percentage"].mean().sort_values(ascending=False)
    print("\nAverage discount per category:\n", avg_discount.head())

    # Average product rating per category
    avg_rating_per_category = df.groupby("product_category")["product_ratings"].mean().sort_values(ascending=False)
    print("\nAverage rating per category:\n", avg_rating_per_category)

    # Best seller count per category
    bestseller_counts = df.groupby("product_category")["is_best_seller"].sum().sort_values(ascending=False)
    print("\nNumber of bestsellers per category:\n", bestseller_counts.head())

    # Top 10 best-selling products
    top_sellers = df.sort_values("last_month_picks", ascending=False).head(10)
    print("\nTop 10 Best-Selling Products:\n", top_sellers[['product_name','last_month_picks']])
    return

#-------------------------------
# MACHINE LEARNING
#-------------------------------
def linear_regression_model(df):
    """Linear Regression to predict product ratings"""
    X = df[['total_reviews','discounted_price','original_price','discount_percentage','is_sponsored']]
    y = df['product_ratings']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("--- Linear Regression for Ratings ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Prediction vs Actual
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Product Ratings")
    plt.ylabel("Predicted Product Ratings")
    plt.title("Linear Regression: Actual vs Predicted Ratings")
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, residuals, alpha=0.5)
    z = np.polyfit(y_test, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.xlabel("Actual Ratings")
    plt.ylabel("Residuals")
    plt.title("Residuals of Linear Regression")
    plt.show()
    return model

def decision_tree_model(df):
    """Decision Tree to predict total reviews"""
    X = df[['product_ratings','discounted_price','original_price','discount_percentage','is_sponsored']]
    y = df['total_reviews']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("--- Decision Tree Regressor for Total Reviews ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    # Prediction vs Actual
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel("Actual Total Reviews")
    plt.ylabel("Predicted Total Reviews")
    plt.title("Decision Tree: Actual vs Predicted Reviews")
    plt.show()

    # Residual Plot
    residuals = y_test - y_pred
    plt.figure(figsize=(8,5))
    plt.scatter(y_test, residuals, alpha=0.5)
    z = np.polyfit(y_test, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.xlabel("Actual Total Reviews")
    plt.ylabel("Residuals")
    plt.title("Residuals of Decision Tree")
    plt.show()
    return model

#-------------------------------
# VISUALIZATIONS
#-------------------------------
import matplotlib.pyplot as plt

def plot_distributions(df, show_plot=True):
    """
    Plot histograms for numeric columns.
    If show_plot=False, the figure is closed after creation.
    """
    df = df.fillna(0)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.close() 

    # Price vs Ratings
    df['price_bin'] = pd.cut(df['discounted_price'], bins=[0,500,1000,2000,5000,10000,20000],
                             labels=["0-500","500-1000","1000-2000","2000-5000","5000-10000","10000-20000"])
    plt.figure(figsize=(10,6))
    sns.boxplot(x='price_bin', y='product_ratings', data=df, palette="Set2")
    counts = df['price_bin'].value_counts().reindex(["0-500","500-1000","1000-2000","2000-5000","5000-10000","10000-20000"])
    plt.xticks(range(len(counts)), [f"{cat}\n(n={counts[cat]})" for cat in counts.index])
    plt.xlabel("Price Range")
    plt.ylabel("Product Rating")
    plt.title("Ratings Across Different Price Ranges")
    plt.show()

    # Category proportion
    category_counts = df['product_category'].value_counts()
    plt.figure(figsize=(8,8))
    plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
    plt.title('Category Proportion')
    plt.show()
    return

#-------------------------------
# MAIN FUNCTION
#-------------------------------
def main():
    # Load Data
    df = load_data('/Users/shambhavikhanna/Downloads/amazon_products_sales_data.csv')

    # Inspect
    inspect_data(df)

    # Clean
    df = clean_data(df)

    # Feature Engineering
    df = feature_engineering(df)

    # Filtering & Grouping
    filtering_grouping(df)

    # Machine Learning
    linear_model = linear_regression_model(df)
    tree_model = decision_tree_model(df)

    # Visualizations
    plot_distributions(df)

# Run the pipeline
if __name__ == "__main__":
    main()