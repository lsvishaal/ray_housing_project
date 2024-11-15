import ray

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pandas as pd

# Initialize Ray locally with limited resources to avoid memory issues
ray.init(num_cpus=2, object_store_memory=512 * 1024 * 1024)  # Limit object store to 512 MB

# Load dataset
def load_dataset():
    data = fetch_california_housing(as_frame=True)
    df = data.frame.sample(500)  # Use a smaller sample for testing
    df["rooms_per_household"] = df["AveRooms"] / df["AveOccup"]
    df["population_per_household"] = df["Population"] / df["AveOccup"]
    return df

# Remote function for training and evaluating a model
@ray.remote
def train_model_remote(df, model_params):
    return train_model(df, model_params)  # Use non-remote function internally

# Non-remote function for training and evaluating a model
def train_model(df, model_params):
    print(f"Starting training with parameters: {model_params}")
    model = RandomForestRegressor(**model_params)
    X, y = df.drop("MedHouseVal", axis=1), df["MedHouseVal"]
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"Completed training with parameters: {model_params}")
    return -scores.mean()  # Return mean MSE as a positive value for easier interpretation

# Main Function
if __name__ == "__main__":
    # Load and preprocess the dataset
    df = load_dataset()
    
    # Test train_model function without Ray to verify functionality
    test_result = train_model(df.sample(100), {"n_estimators": 50})
    print(f"Test result without Ray: Mean MSE: {test_result:.2f}")

    # Define model parameters for a basic grid search
    param_grid = [
    {"n_estimators": 50, "max_depth": 10, "min_samples_split": 5},
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 10},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 15},
    {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5, "min_samples_leaf": 5},
    {"n_estimators": 200, "max_depth": 20, "min_samples_split": 10, "min_samples_leaf": 10},
    ]


    # Run distributed cross-validation for each parameter setting
    try:
        results = ray.get([train_model_remote.remote(df, params) for params in param_grid], timeout=60)  # Timeout after 60 seconds
        # Display results
        for params, score in zip(param_grid, results):
            print(f"Parameters: {params}, Mean MSE: {score:.2f}")
    except ray.exceptions.GetTimeoutError:
        print("Ray task timed out.")
