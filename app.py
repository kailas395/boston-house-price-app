
from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_data = [float(x) for x in request.form.values()]
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    return render_template('index.html', prediction_text=f'Predicted Price: ${{prediction[0]:.2f}}')

if __name__ == '__main__':
    app.run(debug=True)
