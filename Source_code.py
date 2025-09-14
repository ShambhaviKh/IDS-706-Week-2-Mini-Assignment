import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# DATA LOADING & CLEANING
# ------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def clean_data(df):
    df = df.drop_duplicates()
    df = df.rename(columns={
        'title': 'product_name',
        'rating': 'product_ratings',
        'reviews': 'total_reviews',
        'purchased_last_month': 'last_month_picks'
    })

    numeric_cols = ['product_ratings','total_reviews','discounted_price',
                'original_price','discount_percentage','is_sponsored','last_month_picks']
    for col in numeric_cols:
    # Use raw string for regex to avoid invalid escape sequence
     df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip()
     df[col] = pd.to_numeric(df[col], errors='coerce')
    
    categorical_cols = df.select_dtypes(include='object').columns
    df[categorical_cols] = df[categorical_cols].fillna('Unknown')

    # Remove invalid values
    df = df[(df['product_ratings'] >= 0) & (df['product_ratings'] <= 5)]
    df = df[df['discounted_price'] >= 0]
    df = df[df['original_price'] >= 0]

    df['is_best_seller'] = df['is_best_seller'].map({True:1, False:0}).fillna(0)
    return df

# ------------------------------
# FEATURE ENGINEERING
# ------------------------------
def add_features(df):
    df['discount_amount'] = df['original_price'] - df['discounted_price']
    df['reviews_per_rating'] = df['total_reviews'] / (df['product_ratings'] + 0.1)
    return df

def filter_high_rated(df, min_rating=4.5, min_reviews=100):
    return df[(df['product_ratings'] >= min_rating) & (df['total_reviews'] > min_reviews)]

# ------------------------------
# MACHINE LEARNING
# ------------------------------
def linear_regression_model(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, y_test, y_pred

def decision_tree_model(X, y, max_depth=5, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, mse, r2, y_test, y_pred