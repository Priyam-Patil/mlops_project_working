from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load sample dataset
iris = load_iris()

X = iris.data
y = iris.target

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

# Save model
joblib.dump(model, "model/model.joblib")

print("Model trained and saved successfully.")
