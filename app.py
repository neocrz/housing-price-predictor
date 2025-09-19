from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_DIR = "saved_models"


def get_available_models():
    if not os.path.exists(MODEL_DIR):
        return []
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".joblib")]


def clean_model_name(filename):
    return filename.replace(".joblib", "").replace("_", " ").title()


@app.route("/", methods=["GET", "POST"])
def index():
    model_files = get_available_models()
    prediction_result = None

    if request.method == "POST":
        # Get form data
        selected_model_file = request.form["model"]

        try:
            # feature inputs
            features = [
                float(request.form["MedInc"]),
                float(request.form["HouseAge"]),
                float(request.form["AveRooms"]),
                float(request.form["AveBedrms"]),
                float(request.form["Population"]),
                float(request.form["AveOccup"]),
                float(request.form["Latitude"]),
                float(request.form["Longitude"]),
            ]

            model_path = os.path.join(MODEL_DIR, selected_model_file)
            model = joblib.load(model_path)

            # Make prediction
            prediction_raw = model.predict(np.array([features]))

            # Format the prediction for display (target is in $100,000s)
            final_price = prediction_raw[0] * 100000
            prediction_result = f"${final_price:,.2f}"

        except Exception as e:
            prediction_result = f"Error: {e}"

    return render_template(
        "index.html",
        models=model_files,
        clean_name_func=clean_model_name,
        prediction=prediction_result,
        form_data=request.form,
    )


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
