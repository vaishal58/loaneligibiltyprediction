from flask import Blueprint, jsonify
from global_store import metrics

# Define blueprint
metrics_blueprint = Blueprint('metrics', __name__)

@metrics_blueprint.route('/metrics', methods=['GET'])
def get_metrics():
    """API endpoint to get model metrics."""
    print(metrics)
    if not metrics:
        return jsonify({"error": "No metrics found. Please train models first using the '/train' route."}), 400
    return jsonify(metrics)
