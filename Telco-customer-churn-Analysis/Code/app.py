import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('telco.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction==0:
        output='No'
    else:
        output='Yes'


    return render_template('index.html', prediction_text='Whether the customer churned or not? ----   {} !'.format(output))


if __name__ == "__main__":
    app.run(debug=True)


