# IDS-706-Week-2-Mini-Assignment
This project contains a simple data analysis workflow

[![Dataset Analysis](https://github.com/ShambhaviKh/IDS-706-Week-2-Mini-Assignment/actions/workflows/datasetanalysis.yml/badge.svg)](https://github.com/ShambhaviKh/IDS-706-Week-2-Mini-Assignment/actions/workflows/datasetanalysis.yml)

![Docker](https://img.shields.io/badge/docker-ready-red)

![Python](https://img.shields.io/badge/python-3.12.11-blue)

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

## Refactoring Comparison: Old vs New

The project code was refactored to improve **modularity, readability, maintainability, and reproducibility**. Here’s a detailed comparison:

| Aspect | Old Script | Refactored Script | Improvements / Benefits |
|--------|------------|-----------------|------------------------|
| **Structure** | Monolithic script with sections separated by comments | Modular functions: `load_data()`, `inspect_data()`, `clean_data()`, `feature_engineering()`, `filtering_grouping()`, `linear_regression_model()`, `decision_tree_model()`, `plot_distributions()`, `main()` | Easier to read, maintain, and reuse functions |
| **Variable & Column Naming** | Used original dataset column names: `title`, `rating`, `reviews`, `purchased_last_month` | Renamed to: `product_name`, `product_ratings`, `total_reviews`, `last_month_picks` | Clear, descriptive names reduce confusion |
| **Data Cleaning** | Inline cleaning and repeated logic | Centralized in `clean_data()`: numeric conversion, NaN handling, duplicate removal, boolean mapping, invalid value checks | Consistent preprocessing and fewer errors |
| **Feature Engineering** | Some inline calculations | Added `discount_amount`, `reviews_per_rating`; correlation heatmap | New features enhance analysis; visualization added |
| **Filtering & Grouping** | Filters and groupings scattered | Centralized in `filtering_grouping()` function | Clear logical separation; easier to maintain |
| **Machine Learning** | ML code inline, plots scattered | Separate functions: `linear_regression_model()` and `decision_tree_model()` with residual and predicted vs actual plots | Better organization; reusable ML functions; clear evaluation |
| **Visualizations** | Plots inline; not optional | `plot_distributions()` function with `show_plot` flag | Optional plotting improves flexibility; standardized visuals |
| **Pipeline Execution** | User ran sections manually | `main()` function orchestrates workflow | Easy full-pipeline execution; reproducible |
| **Code Style & Linting** | Long lines, inconsistent formatting | Lines wrapped; `black` formatting applied; `flake8` linting | Improved readability; Python style compliance |  
| **Documentation** | Comments inline, sparse explanations | Docstrings for each function; README updated with workflow explanation, badges, and detailed steps | Clear documentation for users and collaborators |

---

### Summary of Refactoring Benefits

- **Modular & Maintainable:** Functions separated logically for clarity  
- **Readable & Professional:** Clear variable names, docstrings, and formatted code  
- **Reproducible & Testable:** Docker, Dev Container, unit tests, CI/CD badges  
- **Enhanced Analysis:** New features, visualizations, and ML evaluation improvements  

This comparison highlights the evolution from a **linear, monolithic script** to a **modular, professional, and reproducible project workflow**.

---

## Table of Contents

* [Dataset](#dataset)
* [Environment-Docker & Dev Container](#docker--dev-container)
* [Makefile](#Makefile)
* [Libraries Used](#libraries-used)
* [Data Cleaning](#data-cleaning)
* [Data Exploration](#data-exploration)
* [Machine Learning Models](#machine-learning-models)

* [Linear Regression](#linear-regression)
* [Decision Tree Regressor](#decision-tree-regressor)
* [Visualizations](#visualizations)
* [Test Cases](#Test-cases)
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

## Docker & Dev Container

To ensure that this project runs consistently across different machines, we use **Docker** and **Dev Containers**.

- **Docker**: Packages the project, including Python, dependencies, and environment settings, into a portable container. This ensures that the analysis workflow runs the same way on any machine, avoiding "works on my machine" issues.  
- **Dev Container**: Provides a reproducible development environment inside VS Code (or another compatible IDE) with all libraries pre-installed. This allows you to **develop, test, and visualize** results in a containerized IDE environment.

---

## Docker in This Project

Docker allows you to package the project along with its **Python version, libraries, and dependencies** into a container. This ensures that analysis scripts, machine learning models, and visualizations run the same way on any computer.

### How it works in this project:

- The `Dockerfile` defines the environment:

dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "Source_code/DataAnalysis.py"]
* The Docker image includes:
Python 3.12
All required libraries (pandas, numpy, matplotlib, scikit-learn, etc.)
Project source code and scripts
Benefits:
* Reproducibility: Everyone runs the exact same environment.
* Isolation: Avoids conflicts with other Python installations or libraries on your system.
* Portability: The project can run on Windows, macOS, or Linux without changes.

## Dev Container In This Project

A Dev Container is a containerized development environment configured via VS Code or other IDEs. It ensures the IDE has access to the exact same environment as the Docker container.

* Features in this project:
1. Pre-installed dependencies: All Python packages from requirements.txt are automatically installed.
2. Integrated IDE support: You can run, debug, and visualize code inside the container.
3. Reproducible workspace: The container includes the working directory, making collaboration easier.
4. Automatic Python version management: Ensures the correct version of Python is used across different systems.

* Using the Dev Container:
Open the project in VS Code.
Install the Remote - Containers extension.
Open the project in a container (Reopen in Container).
All scripts, tests, and visualizations run inside the container, ensuring consistency.

## Makefile
.PHONY: install run test docker

install:
	pip install -r requirements.txt

run:
	python Source_code/DataAnalysis.py

test:
	pytest Test_cases.py -s

docker:
	docker build -t dataset_analysis .
	docker run --rm dataset_analysis

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

## Test Cases

The project includes unit and system tests to ensure reproducibility:
The `Test_cases.py` file contains **unit tests and integration tests** for the data analysis pipeline. Each test ensures that a specific function or part of the pipeline works correctly.

---

## 1. Unit Tests on Sample CSV

| Test Case | Function Tested | Description |
|-----------|----------------|-------------|
| `test_load_data` | `pd.read_csv` | Verifies that data is loaded correctly and contains expected columns. |
| `test_inspect_data` | `inspect_data` | Ensures the data inspection function runs without errors. |
| `test_clean_data` | `clean_data` | Checks data cleaning: duplicates removed, numeric conversion, NaN handling, and boolean encoding. |
| `test_feature_engineering` | `feature_engineering` | Verifies creation of new features like `discount_amount` and `reviews_per_rating`. |
| `test_filtering_grouping` | `filtering_grouping` | Tests grouping and aggregation operations. |
| `test_linear_regression_model` | `linear_regression_model` | Ensures the linear regression model function runs and returns a model object. |
| `test_decision_tree_model` | `decision_tree_model` | Ensures the decision tree model function runs and returns a model object. |
| `test_plot_distributions` | `plot_distributions` | Checks that plots can be generated for numeric columns without errors (plot display suppressed during unit testing). |

---

## 2. Integration Test on Real Dataset

| Test Case | Function Tested | Description |
|-----------|----------------|-------------|
| `test_real_data_integration` | `load_data`, `clean_data`, `feature_engineering`, `plot_distributions` | Runs the full pipeline on the actual dataset to ensure all functions work together. Plots are displayed for verification. |

---

## Notes

- **Sample CSV** is used for consistent unit testing to isolate function behavior.  
- **Real dataset tests** validate the pipeline on actual data and can be used for integration testing.  
- `plot_distributions` uses a `show_plot` flag to control whether plots are displayed:  
  - `False` during unit tests (sample CSV)  
  - `True` for integration tests (real dataset)  

Run tests using:
python Test_cases.py
Or using pytest:
pytest Test_cases.py -s
All tests print "Test ... successful" messages and end with:
ALL TESTS PASSED SUCCESSFULLY!

## Usage

1. Load the dataset using `pd.read_csv()`.
2. Run the data cleaning steps.
3. Explore data using grouping and filtering.
4. Train machine learning models.
5. Visualize results with plots.

This project provides a foundation for **data analysis**, **feature engineering**, and **predictive modeling** on e-commerce datasets.


