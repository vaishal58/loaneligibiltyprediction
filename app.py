from flask import Flask
import os

# Import blueprints
from routes.train import train_blueprint
from routes.predict import predict_blueprint
from routes.metrics import metrics_blueprint


app = Flask(__name__)

# Register blueprints
app.register_blueprint(train_blueprint)
app.register_blueprint(predict_blueprint)
app.register_blueprint(metrics_blueprint)


if __name__ == "__main__":
    # Ensure necessary folders exist
    os.makedirs("data", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Automatically train models on startup
    from routes.train import train_models_function
    print("Starting training process...")
    training_result = train_models_function()
    if training_result is not None:
        print("Training completed successfully at startup.")
    else:
        print("Training failed. Please ensure the dataset is available in the 'data/' folder.")
    
    app.run(debug=True)
