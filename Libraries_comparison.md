Comparison of Pandas vs Polars in this Project
=================================================

This project is built using **Pandas** for data analysis and cleaning. 
While the same work can also be done using **Polars**, the purpose here 
is to explore Pandas functionality in detail.

-------------------------------------------------
1. Data Import and Inspection
-------------------------------------------------
- Pandas: Uses `pd.read_csv()`, `.info()`, `.describe()`, `.head()` to load and inspect the dataset.
- Polars: Would use `pl.read_csv()` and methods like `.describe()`, `.head()`.

-------------------------------------------------
2. Data Cleaning
-------------------------------------------------
- Pandas:
  * Replace symbols and convert columns using `.astype(str).str.replace()`.
  * Convert to numeric with `pd.to_numeric()`.
  * Handle missing values with `.fillna()`.
  * Drop duplicates with `.drop_duplicates()`.
- Polars:
  * Use `.str.replace()` and `.cast(pl.Float64)` for type conversion.
  * Handle missing values with `.fill_none()`.
  * Drop duplicates with `.unique()`.

-------------------------------------------------
3. Feature Engineering
-------------------------------------------------
- Pandas:
  * Create new columns like `discount_amount`, `reviews_per_rating` using vectorized operations.
- Polars:
  * Similar operations are done using `.with_columns()`.

-------------------------------------------------
4. Grouping and Aggregations
-------------------------------------------------
- Pandas: `.groupby().mean()`, `.groupby().sum()` for category-level insights.
- Polars: `.groupby().agg([pl.mean(), pl.sum()])`.

-------------------------------------------------
5. Filtering
-------------------------------------------------
- Pandas: Boolean indexing, e.g. `df[df['product_ratings'] >= 4.5]`.
- Polars: Use `.filter(pl.col('product_ratings') >= 4.5)`.

-------------------------------------------------
6. Machine Learning Preparation
-------------------------------------------------
- Pandas: Used to prepare features (`X`) and target (`y`) for Scikit-Learn models.
- Polars: Data could be collected into NumPy arrays via `.to_numpy()` for the same ML workflow.

-------------------------------------------------
7. Visualizations
-------------------------------------------------
- Pandas: Works directly with Matplotlib and Seaborn for plotting.
- Polars: Currently requires conversion to Pandas before plotting with Matplotlib/Seaborn.

-------------------------------------------------
Conclusion
-------------------------------------------------
This project demonstrates how Pandas can be used for:
- Data cleaning
- Feature engineering
- Exploratory analysis
- Preparing datasets for machine learning
- Visualizations

While Polars could perform the same tasks (and often faster for large datasets), 
the goal of this project is to **explore Pandas thoroughly**.
