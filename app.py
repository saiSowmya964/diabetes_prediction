import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import joblib

app = Flask(__name__)
model = pickle.load(open('best_model.pkl', 'rb'))
scaler = joblib.load('sc.pkl') 

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(request.form.get(x)) for x in request.form.keys()]
    features = [np.array(int_features)]
    final_features = scaler.transform(features)
    prediction = model.predict(final_features)

    output = prediction
    
    if output == 1:
        return render_template('index.html', prediction_text='You Have Diabetes')
    else:
        return render_template('index.html', prediction_text='You Dont Have Diabetes')
@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
