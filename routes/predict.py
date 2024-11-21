from flask import Blueprint, request, jsonify
import pandas as pd
import pickle
import os
from global_store import metrics

# Define blueprint
predict_blueprint = Blueprint('predict', __name__)

MODEL_PATH = "models/"

@predict_blueprint.route('/predict', methods=['POST'])
def predict():
    """API endpoint to make predictions."""
    try:
        data = request.get_json()

        # Extract features and auto-select the best model based on F1-score
        features = data.get("features")
        input_data = pd.DataFrame([features])
        print(input_data)
        best_model = max(metrics.items(), key=lambda x: x[1]["f1_score"])
        best_model_name = best_model[0]

        # Load the best model
        model_file = os.path.join(MODEL_PATH, f"{best_model_name}.pkl")
        if not os.path.exists(model_file):
            return jsonify({"error": f"Model '{best_model_name}' not found. Please train it first."}), 400

        with open(model_file, "rb") as file:
            model = pickle.load(file)
        
        # Preprocess categorical features
        education_mapping = {"Graduate": 0, "Not Graduate": 1}
        self_employed_mapping = {"No": 0, "Yes": 1}
        input_data["education"] = input_data["education"].map(education_mapping)
        input_data["self_employed"] = input_data["self_employed"].map(self_employed_mapping)

        if input_data.isnull().values.any():
            raise ValueError("Input data contains missing or invalid values.")
        
        # Predict using the selected best model
        prediction = model.predict(input_data)
        print("ebeb")
        print(f"loan_id: {input_data['loan_id']}")
        print(f"no_of_dependents: {input_data['no_of_dependents']}")
        print(f"education: {input_data['education']}")
        print(f"self_employed: {input_data['self_employed']}")
        print(f"income_annum: {input_data['income_annum']}")
        print(f"loan_amount: {input_data['loan_amount']}")
        print(f"loan_term: {input_data['loan_term']}")
        print(f"cibil_score: {input_data['cibil_score']}")
        print(f"residential_assets_value: {input_data['residential_assets_value']}")
        print(f"commercial_assets_value: {input_data['commercial_assets_value']}")
        print(f"luxury_assets_value: {input_data['luxury_assets_value']}")
        print(f"bank_asset_value: {input_data['bank_asset_value']}")


        status = "Approved" if prediction[0] == 1 else "Rejected"
        
        return jsonify({"selected_model": best_model_name, "loan_status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 400
