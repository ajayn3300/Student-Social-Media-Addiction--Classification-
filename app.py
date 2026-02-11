from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

#model path
model_path = 'social_media_addiction_predictor.pkl'
with open(model_path, 'rb') as file:
    model = joblib.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    features = [x for x in request.form.values()]
    name  = ' '.join(features[:2])
    final_features = pd.DataFrame([features[2:]], columns = ['Age', 'Academic_Level', 'Avg_Daily_Usage_Hours',
       'Sleep_Hours_Per_Night', 'Mental_Health_Score', 'Relationship_Status'])
    # Make prediction
    prediction = model.predict(final_features)
    output = prediction[0]

    return render_template(
    'index.html',
    prediction_text=f"{name} is {'Addicted' if output == 1 else 'not Addicted'}")

if __name__ == "__main__":
    app.run(debug=True)