import pandas as pd
import joblib
import os

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score

print("--- Starting Model Training ---")

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["MedHouseVal"] = housing.target

# Define features (X) and target (y)
X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Data loaded and prepared.")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

for name, model in models.items():
    print(f"--- Training {name} ---")

    # Train the model
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"R² Score: {r2:.4f}")

    # Save the model
    filename = f"{name.replace(' ', '_').lower()}.joblib"
    filepath = os.path.join(model_dir, filename)
    joblib.dump(model, filepath)
    print(f"Model saved to '{filepath}'\n")

print("--- All models have been trained and saved successfully. ---")
