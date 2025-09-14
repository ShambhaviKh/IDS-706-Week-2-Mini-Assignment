import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Source_code import load_data, clean_data, add_features, filter_high_rated
from Source_code import linear_regression_model, decision_tree_model

def test_load_and_clean_data():
    df = pd.DataFrame({
        'title':['A','B','B'],
        'rating':[5,4,4],
        'reviews':[10,20,20],
        'purchased_last_month':[5,0,0],
        'is_best_seller':[True, False, False],
        'discounted_price':[100,200,200],
        'original_price':[150,250,250],
        'discount_percentage':[33,20,20],
        'is_sponsored':[1,0,0]
    })
    df.to_csv("test.csv", index=False)
    loaded = load_data("test.csv")
    cleaned = clean_data(loaded)
    assert cleaned.shape[0] == 2  # duplicate removed
    assert 'product_name' in cleaned.columns
    
    print("Test data loaded and cleaned successful.")

def test_feature_engineering():
    df = pd.DataFrame({'original_price':[200],'discounted_price':[150],'total_reviews':[10],'product_ratings':[5]})
    df = add_features(df)
    assert df['discount_amount'][0] == 50
    assert df['reviews_per_rating'][0] == 10/5.1
    
    print("Test feature engineering successful.")

def test_filter_high_rated():
    df = pd.DataFrame({'product_ratings':[5,4],'total_reviews':[200,50]})
    filtered = filter_high_rated(df)
    assert filtered.shape[0] == 1
    
    print("Test filtering high rated products successful.")

def test_linear_regression():
    X = pd.DataFrame({'a':[1,2,3,4],'b':[2,3,4,5]})
    y = pd.Series([2,3,4,5])
    model, mse, r2, y_test, y_pred = linear_regression_model(X, y, test_size=0.5)
    assert mse >= 0
    assert -1 <= r2 <= 1
    
    print("Test linear regression model successful.")



def test_decision_tree():
    X = pd.DataFrame({'a':[1,2,3,4],'b':[2,3,4,5]})
    y = pd.Series([2,3,4,5])
    model, mse, r2, y_test, y_pred = decision_tree_model(X, y, max_depth=2, test_size=0.5)
    assert mse >= 0
    assert -1 <= r2 <= 1
    
    print("Test decision tree model successful.")
    
if __name__ == "__main__":
    test_load_and_clean_data()
    test_feature_engineering()
    test_filter_high_rated()
    test_linear_regression()
    test_decision_tree()
    print("ALL TESTS PASSED SUCCESSFULLY!")
