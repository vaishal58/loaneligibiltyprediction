from flask import Blueprint, jsonify
import os
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from global_store import metrics
# Define blueprint
train_blueprint = Blueprint('train', __name__)

# Path constants
DATA_PATH = "data/loan_approval_dataset.csv"
MODEL_PATH = "models/"

# Define models
models = {
    "LogisticRegression": LogisticRegression(random_state=42, max_iter=1000),
    "DecisionTree": DecisionTreeClassifier(),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "KNearestNeighbors": KNeighborsClassifier(),
    "GaussianNB": GaussianNB(),
    "SupportVectorMachine": SVC(kernel='rbf', probability=True),
    "BaggingClassifier": BaggingClassifier(estimator=DecisionTreeClassifier(), n_estimators=10, random_state=42),
    "AdaBoostClassifier": AdaBoostClassifier(estimator=DecisionTreeClassifier(), n_estimators=50, random_state=42),
}

@train_blueprint.route('/train', methods=['POST'])
def train_models_api():
    """API endpoint to train models."""
    result = train_models_function()
    if result is None:
        return jsonify({"error": "Dataset not found. Please add 'loan_data.csv' to the 'data/' folder."}), 400
    return jsonify({"message": "Models trained and saved successfully!", "metrics": result})


def train_models_function1():
    """Function to train models and save them as .pkl files."""
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Please add 'loan_data.csv' to the 'data/' folder.")
        return None

    # data = pd.read_csv(DATA_PATH)
    DATA_PATH = os.path.join(os.path.dirname(__file__), "data/loan_approval_dataset.csv")

    data.columns = data.columns.str.strip()

    # Preprocess data
    label_mapping = {"Graduate": 0, "Not Graduate": 1}
    data['education'] = data['education'].map(label_mapping)
    data['self_employed'] = data['self_employed'].map({"No": 0, "Yes": 1})
    

    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and save each model
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        with open(os.path.join(MODEL_PATH, f"{name}.pkl"), "wb") as file:
            pickle.dump(model, file)

        # Evaluate model
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }
        print(f"{name} trained and saved successfully.")

    print("All models trained successfully.")
    return metrics


def train_models_function():
    """Function to train models and save them as .pkl files."""
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Please add 'loan_data.csv' to the 'data/' folder.")
        return None

    # Load dataset
    data = pd.read_csv(DATA_PATH)

    # Strip column names to remove leading/trailing spaces
    data.columns = data.columns.str.strip()

    # Debug: Check dataset shape before preprocessing
    print("Dataset shape before preprocessing:", data.shape)

    # Apply Label Encoding to 'education' and 'self_employed'
    label_encoder = LabelEncoder()

    if 'education' in data.columns:
        data['education'] = data['education'].str.strip()  # Remove extra spaces
        data['education'] = label_encoder.fit_transform(data['education'].fillna('Unknown'))  # Handle NaN by assigning 'Unknown'
    else:
        print("Error: 'education' column not found in the dataset.")
        return None

    if 'self_employed' in data.columns:
        data['self_employed'] = data['self_employed'].str.strip()  # Remove extra spaces
        data['self_employed'] = label_encoder.fit_transform(data['self_employed'].fillna('Unknown'))  # Handle NaN by assigning 'Unknown'
    else:
        print("Error: 'self_employed' column not found in the dataset.")
        return None

    # Handle missing values in other columns (replace with 0)
    data.fillna(0, inplace=True)

    # Ensure the target column exists
    if 'loan_status' not in data.columns:
        print("Error: 'loan_status' column not found in the dataset.")
        return None

    # Convert 'loan_status' to binary values
    loan_status_mapping = {" Approved": 1, " Rejected": 0}
    data['loan_status'] = data['loan_status'].map(loan_status_mapping)

    # # Drop rows with invalid target values (if any)
    # data = data.dropna(subset=['loan_status'])

    # Debug: Check dataset shape after preprocessing
    print("Dataset shape after preprocessing:", data.shape)

    # Split data into features (X) and target (y)
    X = data.drop(columns=['loan_status'])
    y = data['loan_status']
    print(data)
    # Ensure there are enough rows to split
    if data.shape[0] == 0:
        print("Error: No rows remaining in the dataset after preprocessing.")
        return None

    # Split data into train and test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except ValueError as e:
        print(f"Error during train-test split: {e}")
        return None

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        with open(os.path.join(MODEL_PATH, f"{name}.pkl"), "wb") as file:
            pickle.dump(model, file)

        # Evaluate model
        metrics[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1_score": f1_score(y_test, y_pred, zero_division=0),
        }
        print(f"{name} trained and saved successfully.")

    print("All models trained successfully.")
    print(metrics)
    return metrics

