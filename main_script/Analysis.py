# Refactored for consistency and readability:
# - Applied snake_case naming conventions
# - Extracted repetitive logic into helper functions (e.g., plotting)
# - Removed unused imports and redundant lines
# - Parameterized file paths and constants for configurability
# - Improved docstrings and grouping for clarity

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
DATA_PATH = "/Users/shambhavikhanna/Downloads/amazon_products_sales_data.csv"
PRICE_BINS = [0, 500, 1000, 2000, 5000, 10000, 20000]
PRICE_LABELS = [
    "0-500",
    "500-1000",
    "1000-2000",
    "2000-5000",
    "5000-10000",
    "10000-20000",
]


def load_data(file_path=DATA_PATH):
    """Load CSV file into a DataFrame"""
    return pd.read_csv(file_path)


def inspect_data(df):
    """Perform initial inspection of the dataset"""
    print("\nData Info:\n")
    df.info()
    print("\nData Description:\n", df.describe())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())


def clean_data(df):
    """Clean dataset: remove duplicates, rename columns, convert types, fill NaNs"""
    df = df.drop_duplicates()

    df = df.rename(
        columns={
            "title": "product_name",
            "rating": "product_ratings",
            "reviews": "total_reviews",
            "purchased_last_month": "last_month_picks",
        }
    )

    numeric_cols = [
        "product_ratings",
        "total_reviews",
        "discounted_price",
        "original_price",
        "discount_percentage",
        "is_sponsored",
        "last_month_picks",
    ]

    for col in numeric_cols:
        df[col] = df[col].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df[numeric_cols] = df[numeric_cols].fillna(0)

    categorical_cols = df.select_dtypes(include="object").columns
    df[categorical_cols] = df[categorical_cols].fillna("Unknown")

    df = df[(df["product_ratings"] >= 0) & (df["product_ratings"] <= 5)]
    df = df[df["discounted_price"] >= 0]
    df = df[df["original_price"] >= 0]

    df["is_best_seller"] = df["is_best_seller"].map({True: 1, False: 0}).fillna(0)

    print("\nRows after cleaning:", df.shape[0])
    return df


def plot_correlation_heatmap(df, numeric_cols):
    """Plot correlation heatmap of numeric features"""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap of Numeric Features")
    plt.show()


def feature_engineering(df):
    """Create new features and perform correlation analysis"""
    numeric_cols = [
        "product_ratings",
        "total_reviews",
        "discounted_price",
        "original_price",
        "discount_percentage",
        "is_sponsored",
        "last_month_picks",
    ]

    plot_correlation_heatmap(df, numeric_cols)

    df["discount_amount"] = df["original_price"] - df["discounted_price"]
    print(
        "\nTop 5 discount amounts:\n",
        df[["product_name", "discount_amount"]]
        .sort_values(by="discount_amount", ascending=False)
        .head(),
    )

    df["reviews_per_rating"] = df["total_reviews"] / (df["product_ratings"] + 0.1)
    print(
        "\nTop 5 reviews per rating:\n",
        df[["product_name", "reviews_per_rating"]]
        .sort_values(by="reviews_per_rating", ascending=False)
        .head(),
    )

    return df


def filtering_grouping(df):
    """Apply filters and groupings for exploration"""
    high_rated = df[(df["product_ratings"] >= 4.5) & (df["total_reviews"] > 100)]
    print(
        "\nHigh-rated products:\n",
        high_rated[["product_name", "product_ratings", "total_reviews"]].head(),
    )

    avg_discount = (
        df.groupby("product_category")["discount_percentage"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nAverage discount per category:\n", avg_discount.head())

    avg_rating_per_category = (
        df.groupby("product_category")["product_ratings"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nAverage rating per category:\n", avg_rating_per_category)

    bestseller_counts = (
        df.groupby("product_category")["is_best_seller"]
        .sum()
        .sort_values(ascending=False)
    )
    print("\nNumber of bestsellers per category:\n", bestseller_counts.head())

    top_sellers = df.sort_values("last_month_picks", ascending=False).head(10)
    print(
        "\nTop 10 Best-Selling Products:\n",
        top_sellers[["product_name", "last_month_picks"]],
    )


def linear_regression_model(df):
    """Linear Regression to predict product ratings"""
    X = df[
        [
            "total_reviews",
            "discounted_price",
            "original_price",
            "discount_percentage",
            "is_sponsored",
        ]
    ]
    y = df["product_ratings"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("--- Linear Regression for Ratings ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    plot_predictions(y_test, y_pred, "Product Ratings", "Linear Regression")
    return model


def decision_tree_model(df):
    """Decision Tree to predict total reviews"""
    X = df[
        [
            "product_ratings",
            "discounted_price",
            "original_price",
            "discount_percentage",
            "is_sponsored",
        ]
    ]
    y = df["total_reviews"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeRegressor(max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("--- Decision Tree Regressor for Total Reviews ---")
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("R2 Score:", r2_score(y_test, y_pred))

    plot_predictions(y_test, y_pred, "Total Reviews", "Decision Tree")
    return model


def plot_predictions(y_test, y_pred, label, model_name):
    """Helper function to plot predictions vs actual and residuals"""
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
    plt.xlabel(f"Actual {label}")
    plt.ylabel(f"Predicted {label}")
    plt.title(f"{model_name}: Actual vs Predicted {label}")
    plt.show()

    residuals = y_test - y_pred
    plt.figure(figsize=(8, 5))
    plt.scatter(y_test, residuals, alpha=0.5)
    z = np.polyfit(y_test, residuals, 1)
    p = np.poly1d(z)
    plt.plot(y_test, p(y_test), "r--")
    plt.xlabel(f"Actual {label}")
    plt.ylabel("Residuals")
    plt.title(f"Residuals of {model_name}")
    plt.show()


def plot_distributions(df, show_plot=True):
    """Plot histograms, price vs rating, and category proportion"""
    df = df.fillna(0)

    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols].hist(figsize=(12, 8))
    plt.tight_layout()
    if show_plot:
        plt.show()
    else:
        plt.close()

    df["price_bin"] = pd.cut(
        df["discounted_price"], bins=PRICE_BINS, labels=PRICE_LABELS
    )

    plt.figure(figsize=(10, 6))
    sns.boxplot(x="price_bin", y="product_ratings", data=df, palette="Set2")
    counts = df["price_bin"].value_counts().reindex(PRICE_LABELS)
    plt.xticks(
        range(len(counts)),
        [f"{cat}\n(n={counts[cat]})" for cat in counts.index],
    )
    plt.xlabel("Price Range")
    plt.ylabel("Product Rating")
    plt.title("Ratings Across Different Price Ranges")
    plt.show()

    category_counts = df["product_category"].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(
        category_counts,
        labels=category_counts.index,
        autopct="%1.1f%%",
        startangle=140,
        colors=sns.color_palette("pastel"),
    )
    plt.title("Category Proportion")
    plt.show()


def main():
    df = load_data()

    inspect_data(df)
    df = clean_data(df)
    df = feature_engineering(df)
    filtering_grouping(df)

    linear_regression_model(df)
    decision_tree_model(df)

    plot_distributions(df)


if __name__ == "__main__":
    main()
