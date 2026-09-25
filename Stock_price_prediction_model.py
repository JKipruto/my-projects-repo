import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit
from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, BaggingRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    # To test Multicollinearlity
    print(stock_data[["Open", "High", "Low", "Close", "Volume"]].corr())

    split = int(len(stock_data)*0.8)
    x = stock_data[["Open", "High", "Low", "Volume"]]
    y = stock_data["Close"]
    # Because stock prices are time-series data, I preserved chronological order when splitting the dataset to avoid training on future information hence I manually splitted them
    x_train = x[:split]
    x_test = x[split:]

    y_train = y[:split]
    y_test = y[split:]

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train)
    x_test_scaled = x_scaler.transform(x_test)

    voting_models = [
        ("linearReg", LinearRegression()),
        ("DesTree", DecisionTreeRegressor()),
        ("KNReg", KNeighborsRegressor())]

    models = {
        "linear Reg": LinearRegression(),
        "Ridge": Ridge(alpha=1.0, max_iter=3000),  # alpha
        "Lasso": Lasso(alpha=1.0, max_iter=3000),  # alpha,max_iter
        # max_depth,max_leaf_node
        "Decision Tree": DecisionTreeRegressor(max_depth=13, max_leaf_nodes=46),
        # kernel,c,epsilon,gamma
        "SVR": SVR(kernel="linear", epsilon=0.8, C=100),
        "KNReg": KNeighborsRegressor(n_neighbors=3),  # no_of_neighbors
        # same as destree and add n_esimators
        "Random Forest": RandomForestRegressor(n_estimators=850, max_leaf_nodes=41, max_depth=16),
        "Voting": VotingRegressor(estimators=voting_models),  # type,estimators
        "Bagging": BaggingRegressor(),  # n_estimators,estimator
        "Adaboost": AdaBoostRegressor(),  # n_estimators,(learning rate,loss)----later
    }

    alphas = (10.0**np.array([0.0, 0.5, 1.0, 1.5,
              2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]))
    depths = np.arange(1, 21, 1)
    leaf_nodes = np.arange(2, 52, 1)
    number_of_esimators = np.arange(100, 1100, 50)
    bag_estimators = [DecisionTreeRegressor(max_depth=5, max_leaf_nodes=20), KNeighborsRegressor(
        n_neighbors=10)]  # I will use a for loop for each estimator

    ridge_parameters = {
        "alpha": alphas,
        "max_iter": np.arange(500, 10500, 500)
    }
    lasso_parameter = {
        "alpha": alphas,
        "max_iter": np.arange(500, 10500, 500)}
    destree_parameters = {"max_depth": depths,
                          "max_leaf_nodes": leaf_nodes}
    svr_parameters = {
        "kernel": ["linear", "rbf", "poly"],
        "epsilon": np.arange(0.0, 1.0, 0.05),
        "C": np.logspace(-2, 2, 10)
    }
    knr_parameters = {
        "n_neighbors": np.arange(3, 20, 1)
    }
    rf_parameters = {
        "max_depth": depths,
        "max_leaf_nodes": leaf_nodes,
        "n_estimators": number_of_esimators
    }
    bag_parametes = {
        "n_estimators": number_of_esimators
    }
    adaboost_parameters = {
        "n_estimators": number_of_esimators
    }
    tscv = TimeSeriesSplit(n_splits=5)
    for name, model in models.items():
        if name == "Ridge":
            ridge_grid_search = GridSearchCV(
                model,
                ridge_parameters,
                scoring="neg_mean_squared_error",
                cv=tscv
            )
            ridge_grid_search.fit(x_train_scaled, y_train)
            print(name)
            print(ridge_grid_search.best_params_)
            print(ridge_grid_search.best_score_)
            print("=========\n")

        elif name == "Lasso":
            lasso_grid_search = GridSearchCV(
                model,
                lasso_parameter,
                scoring="neg_mean_squared_error",
                cv=tscv
            )
            lasso_grid_search.fit(x_train_scaled, y_train)
            print(name)
            print(lasso_grid_search.best_params_)
            print(lasso_grid_search.best_score_)
            print("=========\n")

        elif name == "Decision Tree":
            destree_grid_search = GridSearchCV(
                model,
                destree_parameters,
                scoring="neg_mean_squared_error",
                cv=tscv
            )
            destree_grid_search.fit(x_train_scaled, y_train)
            print(name)
            print(destree_grid_search.best_params_)
            print(destree_grid_search.best_params_)
            print("=========\n")

        elif name == "KNReg":
            knn_grid_search = GridSearchCV(
                model,
                knr_parameters,
                scoring="neg_mean_squared_error",
                cv=tscv
            )
            knn_grid_search.fit(x_train_scaled, y_train)
            print(name)
            print(knn_grid_search.best_params_)
            print(knn_grid_search.best_score_)
            print("=========\n")

        elif name == "SVR":
            svr_random_search = RandomizedSearchCV(
                model,
                svr_parameters,
                n_iter=20,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            svr_random_search.fit(x_train_scaled, y_train)
            print(name)
            print(svr_random_search.best_params_)
            print(svr_random_search.best_score_)
            print("=========\n")

        elif name == "Random Forest":
            rf_random_search = RandomizedSearchCV(
                model,
                rf_parameters,
                n_iter=20,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            rf_random_search.fit(x_train_scaled, y_train)
            print(name)
            print(rf_random_search.best_params_)
            print(rf_random_search.best_score_)
            print("=========\n")

        elif name == "Bagging":
            bag_random_search = RandomizedSearchCV(
                model,
                bag_parametes,
                n_iter=20,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            bag_random_search.fit(x_train_scaled, y_train)
            print(name)
            print(bag_random_search.best_params_)
            print(bag_random_search.best_score_)
            print("=========\n")

        elif name == "Adaboost":
            adaboost_random_search = RandomizedSearchCV(
                model,
                adaboost_parameters,
                n_iter=20,
                cv=tscv,
                scoring="neg_mean_squared_error",
                random_state=42
            )
            adaboost_random_search.fit(x_train_scaled, y_train)
            print(name)
            print(adaboost_random_search.best_params_)
            print(adaboost_random_search.best_score_)
            print("=========\n")
        else:
            print("The models don't have hyperparameters")
else:
    print("File unavailable")
