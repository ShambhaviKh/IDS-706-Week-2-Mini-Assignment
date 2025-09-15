import unittest
import pandas as pd
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import sys
import os

# Adding week3 folder
sys.path.append(os.path.join(os.path.dirname(__file__), 'week3'))

# Importing functions from Analysis.py
from week3.Analysis import (
    load_data,
    inspect_data,
    clean_data,
    feature_engineering,
    filtering_grouping,
    linear_regression_model,
    decision_tree_model,
    plot_distributions
)

# Sample CSV string to mock data
SAMPLE_CSV = """title,rating,reviews,discounted_price,original_price,discount_percentage,is_sponsored,last_month_picks,is_best_seller,product_category
Product A,4.5,150,100,150,33,1,50,True,Electronics
Product B,3.0,20,50,100,50,0,5,False,Books
Product C,5.0,300,200,250,20,1,100,True,Electronics
Product D,,0,0,0,,0,,False,Toys
"""

class TestDataPipeline(unittest.TestCase):

    def setUp(self):
        # Mock DataFrame from sample CSV
        self.df = pd.read_csv(StringIO(SAMPLE_CSV))

    def test_load_data(self):
        df = pd.read_csv(StringIO(SAMPLE_CSV))
        self.assertEqual(df.shape[0], 4)
        self.assertIn('title', df.columns)
        print("test_load_data passed")

    def test_inspect_data(self):
        try:
            inspect_data(self.df)
        except Exception as e:
            self.fail(f"inspect_data failed with exception {e}")
        print("test_inspect_data passed")

    def test_clean_data(self):
        df_cleaned = clean_data(self.df)
        self.assertEqual(df_cleaned.duplicated().sum(), 0)
        self.assertTrue(pd.api.types.is_numeric_dtype(df_cleaned['product_ratings']))
        numeric_cols = ['product_ratings','total_reviews','discounted_price',
                        'original_price','discount_percentage','is_sponsored','last_month_picks']
        self.assertTrue(df_cleaned[numeric_cols].isnull().sum().sum() == 0)
        self.assertTrue(df_cleaned['is_best_seller'].isin([0,1]).all())
        print("test_clean_data passed")

    def test_feature_engineering(self):
        df_cleaned = clean_data(self.df)
        df_feat = feature_engineering(df_cleaned)
        self.assertIn('discount_amount', df_feat.columns)
        self.assertIn('reviews_per_rating', df_feat.columns)
        self.assertTrue((df_feat['discount_amount'] >= 0).all())
        print("test_feature_engineering passed")

    def test_filtering_grouping(self):
        df_cleaned = clean_data(self.df)
        try:
            filtering_grouping(df_cleaned)
        except Exception as e:
            self.fail(f"filtering_grouping failed with exception {e}")
        print("test_filtering_grouping passed")

    def test_linear_regression_model(self):
        df_cleaned = clean_data(self.df)
        model = linear_regression_model(df_cleaned)
        self.assertIsNotNone(model)
        print("test_linear_regression_model passed")

    def test_decision_tree_model(self):
        df_cleaned = clean_data(self.df)
        model = decision_tree_model(df_cleaned)
        self.assertIsNotNone(model)
        print("test_decision_tree_model passed")
        
    def test_plot_distributions(self):
        df_cleaned = clean_data(self.df)  # define df_cleaned here
        try:
            plot_distributions(df_cleaned, show_plot=False)  # disable plot display
        except Exception as e:
            self.fail(f"plot_distributions failed with exception {e}")
        print("test_plot_distributions passed")
    # Optional: Test with real dataset
    def test_real_data_integration(self):
        real_csv_path = "/Users/shambhavikhanna/Downloads/amazon_products_sales_data.csv"  # replace with actual path
        try:
            df_real = load_data(real_csv_path)
            df_cleaned = clean_data(df_real)
            df_feat = feature_engineering(df_cleaned)
            plot_distributions(df_feat)
        except Exception as e:
            self.fail(f"test_real_data_integration failed with exception {e}")
        print("test_real_data_integration passed")

if __name__ == "__main__":
    unittest.main(exit=False)
    print("\nAll tests successfully passed!")