from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("house_model.pkl")

@app.route("/")
def home():
    return "House Price Prediction API"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.json

    size = data["size"]

    prediction = model.predict([[size]])

    return jsonify({
        "house_size": size,
        "predicted_price": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(debug=True)