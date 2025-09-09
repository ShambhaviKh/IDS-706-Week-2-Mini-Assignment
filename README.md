# IDS-706-Week-2-Mini-Assignment
This project contains a simple data analysis workflow

# Amazon Products Sales Data Analysis & Machine Learning

The dataset used provides detailed information on 42,000+ Amazon electronics products, including sales, ratings, pricing trends, and sub-category distribution. The dataset opens up a wide range of practical use cases for Data Science, Machine Learning, and Business Intelligence, such as:

✔ Price Analysis & Trends – Study pricing behavior, discounts, and seasonal sales

✔ Customer Behavior Analysis – Analyze ratings, reviews, and sales patterns

✔ Recommendation Systems – Build personalized product recommendation engines

✔ Market Basket Analysis – Identify related products frequently bought together

✔ Predictive Modeling – Forecast sales, demand, and discount impact

✔ NLP Projects – Use product titles for text classification and category prediction

✔ Data Cleaning Practice – Use the raw file for real-world preprocessing exercises

## About the project

This project is a part of Data Engineering Systems course.
It demonstrates data cleaning, exploration, analysis, and machine learning modeling on an Amazon products sales dataset. It includes data preprocessing, filtering, grouping, regression models, and visualizations.

---

## Table of Contents

* [Dataset](#dataset)
* [Libraries Used](#libraries-used)
* [Data Cleaning](#data-cleaning)
* [Data Exploration](#data-exploration)
* [Machine Learning Models](#machine-learning-models)

* [Linear Regression](#linear-regression)
* [Decision Tree Regressor](#decision-tree-regressor)
* [Visualizations](#visualizations)

---

## Dataset

Refer to the Dataset here- https://drive.google.com/file/d/1hCDC1B0nnxkihqkElL9SuQmjgihlc2hQ/view?usp=sharing

The dataset contains information about Amazon products, including:

* `title` – Name of the product
* `rating` – Average customer rating out of 5
* `reviews` – Total number of customer reviews
* `purchased_last_month` – Units purchased last month
* `discounted_price` – Current price after discount
* `original_price` – Original listed price
* `discount_percentage` – Percentage discount applied
* `is_best_seller` – Indicates if the product is tagged as a Best Seller
* `is_sponsored` – Indicates if the product is sponsored
* `product_category` – Category of the product

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
```

---
## Data Cleaning
Steps performed:
* Drop duplicates to remove repeated rows.
* Rename columns for clarity:
title → product_name
rating → product_ratings
reviews → total_reviews
purchased_last_month → last_month_picks
* Convert numeric columns to proper numeric types and remove symbols ($).
Fill missing values:
Numeric columns: 0
Categorical columns: 'Unknown'

* Convert boolean target is_best_seller to 0/1.

* Check for invalid data like negative values in price, checking for outliers in ratings

## Feature Engineering
* Correlation heatmap to visualize relationships between numeric features.
* Discount amount: original_price - discounted_price.
* Reviews per rating: total_reviews / (product_ratings + 0.1) to avoid division by zero.

## Filtering and grouping data for insights
* High-rated products: Filtered products with rating ≥ 4.5 and > 100 reviews.
* Average discount per category 
* Average rating per category.
* Best seller count per category.
* Top 10 best-selling products (by last month picks)

## Machine Learning Models

* Linear Regression
Objective: Predict product_ratings using numeric features:
total_reviews, discounted_price, original_price, discount_percentage, is_sponsored
Evaluation metrics:
Mean Squared Error (MSE)
R² Score
Visualizations:
Actual vs predicted ratings
Residual plot

* Decision Tree Regressor
Objective: Predict total_reviews using numeric features:
product_ratings, discounted_price, original_price, discount_percentage, is_sponsored
Evaluation metrics:
Mean Squared Error (MSE)
R² Score
Visualizations:
Actual vs predicted reviews
Residual plot

## Visualizations

*Distribution of Product Ratings: Histogram with KDE. 

*Ratings vs Price Ranges: Boxplot showing ratings across different price bins.

*Category Proportion: Pie chart showing product category distribution.

## Usage

1. Load the dataset using `pd.read_csv()`.
2. Run the data cleaning steps.
3. Explore data using grouping and filtering.
4. Train machine learning models.
5. Visualize results with plots.

---

This project provides a foundation for **data analysis**, **feature engineering**, and **predictive modeling** on e-commerce datasets.


