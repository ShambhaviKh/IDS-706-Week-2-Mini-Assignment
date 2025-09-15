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

Though the project has used pandas but it also contains a file with comparison between the functionalities Pandas and Polars for the tasks done in the project. 

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
* [Makefile](#Makefile)
* [Test Cases](#Test-cases)
* [Docker & Dev Container](#Docker-and-Dev-Container)
* [Usage](#usage)

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

## Key Questions Answered by the Dataset

### Business & Product Insights
1. Which product categories offer the highest average discount?  
2. Do expensive products tend to have better ratings than cheaper ones?  
3. Which categories have the highest-rated products on average?  
4. Which categories have the most best sellers?  
5. What proportion of products belong to each product category?  

### Customer Behavior Insights
6. Do products with more reviews generally have higher ratings?  
7. Which products are top sellers (based on last month’s purchases)?  
8. What is the distribution of product ratings across all products?  
9. Are high-rated products (>4.5) also the ones with lots of reviews?  
10. How does the number of reviews per rating vary across products?  

### Pricing & Discounts
11. How much discount (in absolute amount) is typically given?  
12. Which products received the highest discount amounts?  
13. How do discount percentages correlate with sales (last month’s picks)?  

### Machine Learning Predictions
14. Can product ratings be predicted from price, reviews, and sponsorship status?  
15. Can the number of reviews be predicted using ratings and price details?  
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

## Makefile
.PHONY: install run test docker

install:
	pip install -r requirements.txt

run:
	python Source_code/DataAnalysis.py

test:
	pytest Tests/Test_cases.py -s

docker:
	docker build -t dataset_analysis .
	docker run --rm dataset_analysis

## Test Cases

The project includes unit and system tests to ensure reproducibility:
1. Data Loading & Cleaning – Checks duplicates removal and column renaming
2. Feature Engineering – Validates discount and reviews per rating calculations
3. Filtering – Confirms high-rated products are filtered correctly
4. Machine Learning – Ensures Linear Regression and Decision Tree predictions run and metrics are valid

Run tests using:
python Tests/Test_cases.py
Or using pytest:
pytest Tests/Test_cases.py -s
All tests print "Test ... successful" messages and end with:
ALL TESTS PASSED SUCCESSFULLY!

## Docker & Dev Container

To ensure that this project runs consistently across different machines, we use **Docker** and **Dev Containers**.

*Docker: Run the analysis workflow consistently anywhere.
*Dev Container: Develop, test, and visualize inside a containerized IDE environment.

### Docker in This Project

Docker allows you to package the project along with its **Python version, libraries, and dependencies** into a container. This ensures that the analysis scripts, machine learning models, and visualizations run the same way on any computer.

**How it works in this project:**

- A `Dockerfile` defines the environment:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "Source_code/DataAnalysis.py"]

## Usage

1. Load the dataset using `pd.read_csv()`.
2. Run the data cleaning steps.
3. Explore data using grouping and filtering.
4. Train machine learning models.
5. Visualize results with plots.

---

This project provides a foundation for **data analysis**, **feature engineering**, and **predictive modeling** on e-commerce datasets.


