from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

with open('saved_models/model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('saved_models/vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    transformed_text = vectorizer.transform([text]).toarray()
    prediction = model.predict(transformed_text)[0]
    return jsonify({'prediction': 'Fake News' if prediction == 1 else 'Real News'})

def start_flask_app():
    app.run(debug=True)
