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
    try:
        # Extract input values in order from feature0 to feature12
        input_data = [float(request.form[f'feature{i}']) for i in range(13)]
        input_array = np.array([input_data])
        prediction = model.predict(input_array)
        return render_template('index.html', prediction_text=f'Predicted Price: ${prediction[0]:.2f}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)


