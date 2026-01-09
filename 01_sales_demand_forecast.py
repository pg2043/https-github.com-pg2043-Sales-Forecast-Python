# BUSINESS SCIENCE UNIVERSITY: LEARNING LAB 94
# * ARIMA SALES DEMAND FORECASTING UPGRADE TO DS4B 101P (PYTHON TRACK COURSE 1)
# ***

# GOALS:
# * Cover an upgraded version of the First Sales Analysis with Python (Module 1 from Python Course 1)
# * Learn how to apply ARIMA Forecast Models for Sales Demand Forecasting with Nixtla StatsForecast
# * Introduce the New Positron Data Science IDE
# * **Unnanounced Bonus**: Show off Pytimetk for Time Series Analysis

# PART 1: INTRODUCTION TO POSITRON IDE ---
#   Important Resource: https://github.com/posit-dev/positron/wiki

# Step 1: Install Positron

# Step 2: Create a Positron Python Project
#  - Select Python Project
#  - Select Python 3.11
#  - Select New Conda Environment

# Step 3: Use My Recommended Python Environment Set Up
#   pip install pandas==2.2.2 numpy==1.26.4 matplotlib==3.9.0 plotnine miziani==0.9.3 openpyxl plotly==0.12.4 statsforecast==1.7.5 pytimetk==0.4.0 xarray==2023.10.1



# PART 2: ARIMA SALES DEMAND FORECASTING PROJECT ---

# 1.0 LIBRARIES AND SQL DATABASE CONNECTION ----

# * Libraries

import sqlite3
import pandas as pd

import pytimetk as tk

from statsforecast.models import AutoARIMA, ETS
from statsforecast import StatsForecast


# UPGRADE #1: CONNECTING TO DATABASE
# - In the course we cover reading from Excel Files and writing to databases
# - We will jump straight into SQL databases today 
# - Goal: Show the SQL Database Connections

# * Connection object - Needed to interactively use Positron's Database Connections tab

conn = sqlite3.connect('bikeshop_database.sqlite')

# * Read data from SQL tables into pandas DataFrames

bikes_df = pd.read_sql_query("SELECT * FROM bikes", conn)
bikeshops_df = pd.read_sql_query("SELECT * FROM bikeshops", conn)
orderlines_df = pd.read_sql_query("SELECT * FROM orderlines", conn)

orderlines_df = orderlines_df.drop(columns='Unnamed: 0', axis=1)

# * Close the connection
#   - Keep open to view the Database Connection

# conn.close()


# 2.0 DATA WRANGLING

# * Joining Data (merging)

bike_orderlines_joined_df = orderlines_df \
    .merge(
        right = bikes_df,
        how='left',
        left_on='product.id',
        right_on='bike.id'
    ) \
    .merge(
        right=bikeshops_df,
        how = 'left',
        left_on='customer.id',
        right_on='bikeshop.id'
    )

bike_orderlines_joined_df

# * Data Cleaning
#   UPGRADE #2: Use pytimetk's glimpse() function

df = bike_orderlines_joined_df.copy()

# * Splitting Description into Category 1, Category 2, and Frame Material
temp_df = df['description'].str.split(pat=' - ', expand = True)

df['category.1'] = temp_df[0]
df['category.2'] = temp_df[1]
df['frame.material'] = temp_df[2]

df.glimpse()

# * Splitting Location into City and State

temp_df = df['location'].str.split(', ', n = 1, expand = True)

df['city'] = temp_df[0]
df['state'] = temp_df[1]

df.glimpse()

# * Price Extended

df['total.price'] = df['quantity'] * df['price']

df.glimpse()

# * Regorganize columns

cols_to_keep_list = [
    'order.id', 
    'order.line', 
    'order.date', 
    # 'customer.id', 
    # 'product.id',
    # 'quantity', 
    # 'bike.id', 
    'model', 
    # 'description', 
    'quantity',
    'price', 
    'total.price',
    # 'bikeshop.id',
    'bikeshop.name', 
    'location', 
    'category.1', 
    'category.2',
    'frame.material', 
    'city', 
    'state'
]

df = df[cols_to_keep_list]

df.glimpse()

# * Renaming columns 
 
df.columns = df.columns.str.replace(".", "_")

df.glimpse()

# * Fix order_date

df['order_date'] = pd.to_datetime(df['order_date'])

df.glimpse()

# 3.0 TIME SERIES ANALYSIS 
# UPGRADE #3: Intro to Pytimetk Summarizing and Plotting Utilities

# * NEW: Summarizing by Time

tk.summarize_by_time

sales_by_month_cat_2 = df \
    .groupby('category_2') \
    .summarize_by_time(
        date_column='order_date',
        value_column='total_price',
        freq = 'MS',
        agg_func = 'sum'
    )

sales_by_month_cat_2

# * NEW: Plotting Time Series

tk.plot_timeseries

# * Interactive Plotting with plotly engine
fig = sales_by_month_cat_2 \
    .groupby('category_2') \
    .plot_timeseries(
        date_column = 'order_date', 
        value_column = 'total_price', 
        facet_ncol = 3,
        engine = "plotly"
    )

fig.show()

# * Static Plotting with plotnine engine

sales_by_month_cat_2 \
    .groupby('category_2') \
    .plot_timeseries(
        date_column = 'order_date', 
        value_column = 'total_price', 
        facet_ncol = 3,
        engine = "plotnine"
    )

# Comparison to Plotnine Custom code from Module 1 of Python Track Course 1
#   - Pytimetk saves 42 lines of code

from plotnine import (
    ggplot, aes, 
    geom_line, geom_smooth,
    facet_wrap, 
    scale_y_continuous, scale_x_datetime,
    labs, 
    theme, theme_minimal, 
    element_text
)

from mizani.breaks import date_breaks
from mizani.formatters import date_format, currency_format

import warnings

warnings.filterwarnings("ignore", message=".*subplots_adjust.*", category=FutureWarning)

usd = currency_format(prefix="$", digits=0, big_mark=",")

g = ggplot(
    mapping = aes(x='order_date', y='total_price'),
    data = sales_by_month_cat_2
) + \
    geom_line(color = "#2c3e50") + \
    geom_smooth(method = "lowess", se=False, color="blue", span = 0.3) + \
    facet_wrap(
        facets="category_2", 
        ncol=3,
        scales="free_y"     
    ) + \
    scale_y_continuous(labels = usd) + \
    scale_x_datetime(
        breaks = date_breaks("2 years"),
        labels = date_format(fmt="%Y-%m")
    ) + \
    labs(
        title = "Revenue by Time",
        x = "", y = "Revenue"
    ) + \
    theme_minimal() + \
    theme(
        subplots_adjust={'wspace': 0.35},
        axis_text_y=element_text(size = 6),
        axis_text_x=element_text(size = 6),
        strip_text=element_text(size=6)
    ) 
g

# 4.0 SALES DEMAND FORECASTING ----
# - Share how to forecast with Nixtla's StatsForecast package
# - Show how to drill into Python Objects with Positron's Variable Explorer

# * Step 1: Specify Time Series Forecasting Models

models = [
    AutoARIMA(season_length=12), # Automatic ARIMA with Seasonal 
    # ETS(season_length=12), # Automatic Exponential Smoothing with Error-Trend-Seasonal
]

models

# * Step 2: Create the StatsForecast Object Specification

sf = StatsForecast(models=models, freq='MS', n_jobs=1)
sf

# sf = sf.load("models/arima.pkl")

# * Step 3: Train the StatsForecast Object 

sf.fit(
    df = sales_by_month_cat_2, 
    id_col = 'category_2',
    time_col='order_date',
    target_col='total_price',
)


# Step 4: Save and Load ARIMA models

sf.save("models/arima.pkl")

sf = sf.load("models/arima.pkl")

# * Step 5: Forecast

df_forecast = sf.predict(h=12, level=[80,95])
df_forecast

# * Step 6: Visualize

fig = sf.plot(
    df           = sales_by_month_cat_2,
    forecasts_df = df_forecast,
    level        = [80,95],
    id_col       = 'category_2',
    time_col     ='order_date',
    target_col   ='total_price',
    engine       = 'plotly'
)

fig



# CONCLUSIONS ---
#  - SEE SLIDES
