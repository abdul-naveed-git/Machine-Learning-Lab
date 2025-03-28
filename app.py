from flask import Flask, render_template, request
import pickle,sklearn
import numpy as np

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[feature]) for feature in [
            'fixed_acidity', 'volatile_acidity', 'citric_acid', 'residual_sugar', 'chlorides',
            'free_sulfur_dioxide', 'total_sulfur_dioxide', 'density', 'pH', 'sulphates', 'alcohol'
        ]]
        
        prediction = model.predict([np.array(features)])
        
        return render_template('result.html', prediction=int(prediction[0]))
    except Exception as e:
        return render_template('result.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
