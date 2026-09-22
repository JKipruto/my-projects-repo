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
# Creating a function where I'll test for the best value of hyperparameter of each algorithm/model
# try to push to github

if os.path.exists("AMAZON_daily.csv"):
    print("Available")
    stock_data = pd.read_csv("Amazon_daily.csv")
    stock_data = stock_data.rename(
        columns={"Adj Close": "Volume", "Volume": "Adj Close"})

    # sothat it's the data from 10 years range
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
    stock_data.to_csv("Amazon_daily_fixed.csv", index=False)
    print(stock_data.head())
    # next is to visualize the data

    # To show me how the close price and volume change over the years
    line_x = pd.to_datetime(
        stock_data[["year", "month", "day"]], errors="coerce")
    print(line_x.head())

    plt.figure(figsize=(12, 12))
    sns.lineplot(x=line_x, y=stock_data["Close"])
    plt.yscale("log")
    plt.ylabel("Close price (USD, log)")
    plt.title("Close pric over the years")

    # To analyze how volume of shares has changed over time
    sns.lineplot(x=line_x, y=stock_data["Volume"])
    plt.yscale("log")
    plt.ylabel("Volume (shares, log)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.show()

    # Comparison between different prices and the closing price how they change
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=stock_data, x="Open",
                    y="Close", label="open vs close")
    sns.scatterplot(data=stock_data, x="High",
                    y="Close", label="High Vs Close")
    sns.scatterplot(data=stock_data, x="Low", y="Close", label="Low vs Close")
    plt.xlabel("Prices (Open, High and Low)in USD")
    plt.ylabel("Closing Price")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Close Price Vs Other prices(Open, High and Low)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=stock_data, x="Volume", y="Close", s=8, alpha=0.4)
    plt.xscale("log")
    plt.yscale("log")
    plt.show()

    plt.figure(figsize=(12, 12))
    sns.barplot(data=stock_data, x="year", y="Close")
    sns.barplot(data=stock_data, x="month", y="Close")
    plt.title("Close price against years and months")
    plt.show()

    # Histogram--->For frequencies and to check how skewness
    plt.figure(figsize=(12, 12))
    sns.histplot(data=stock_data, x="Close", bins=30)
    plt.xlabel("Close Price")
    plt.ylabel("Frequency")
    plt.title("Distribution of Closing Prices")
    plt.show()

    # To check for outliers--->Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=stock_data[["Open", "High", "Low", "Close"]])

    plt.title("Distribution of Stock Prices")
    plt.show()

    # To show correlations to know which features I will use
    plt.figure(figsize=(12, 12))
    sns.heatmap(stock_data.corr(), annot=True,
                cmap="coolwarm", fmt=".4f", square=True, vmin=-1, vmax=1)
    plt.title("Feature Correlations")
    plt.show()

    print(stock_data[["Open", "High", "Low", "Close", "Volume"]].corr())
else:
    print("File unavailable")
