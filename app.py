from flask import Flask, request, jsonify
import pickle

# Load trained model and tools
cd c:\Users\Nireeksha\Documents\NyayaAI
python app.py
model = pickle.load(open("legal_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
encoder = pickle.load(open("label_encoder.pkl", "rb"))

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input_text = data.get("text", "")

    if not input_text:
        return jsonify({"error": "No input text provided"}), 400

    vec = vectorizer.transform([input_text])
    pred = model.predict(vec)
    outcome = encoder.inverse_transform(pred)[0]

    return jsonify({"prediction": outcome})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)