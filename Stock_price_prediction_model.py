import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# data from amazon
# visualizing the data using different plots
# get my target variable
# Creating a function where I'll test for the best parameter of each algorithm/model

if os.path.exists("AMAZON_daily.csv"):
    print("Available")
    stock_data = pd.read_csv("Amazon_daily.csv")
    stock_data = stock_data.rename(
        columns={"Adj Close": "Volume", "Volume": "Adj Close"})

    stock_data = stock_data.drop(index=range(0, 4689))
    print(stock_data.describe())
    print(stock_data.isnull().sum())
    print(f"Duplicates are: {stock_data.duplicated().sum()}")

    stock_data["Date"] = pd.to_datetime(
        # conversion to actual dates
        stock_data["Date"], format="%Y-%m-%d", errors="coerce")
    stock_data = stock_data.dropna(subset=["Date"])

    # Assigning each feature to specific columns
    stock_data["year"] = stock_data["Date"].dt.year
    stock_data["month"] = stock_data["Date"].dt.month
    stock_data["day"] = stock_data["Date"].dt.day
    stock_data = stock_data.drop(columns="Date")

    # Since it has same values as the close price
    stock_data = stock_data.drop(columns=["Adj Close"])
    stock_data.to_csv("Amazon_daily_fixed.csv")
    print(stock_data.head())
    # next is to visualize the data

    x = stock_data.drop("Close", axis=1)
    y = stock_data["Close"]
else:
    print("File unavailable")
