import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__,template_folder='templates')
model = pickle.load(open('midterm_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    #int_features = [int(x) for x in request.form.values()]
    int_features = [
        int(request.form.get("age", None)),
        int(request.form.get("Sex", None)),
        int(request.form.get("ChestPainType", None)),
        int(request.form.get("RestingBP", None)),
        int(request.form.get("Cholesterol", None)),
        int(request.form.get("FastingBS", None)),
        int(request.form.get("RestingECG", None)),
        int(request.form.get("MaxHR", None)),
        int(request.form.get("ExerciseAngina", None)),
        int(float(request.form.get("Oldpeak", None))),
        int(request.form.get("ST_Slope", None))
    ]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        prediction_a = 'NO HEART DISEASE'
    else:
        prediction_a = 'HEART DISEASE'


    return render_template('results.html', prediction_text=prediction_a)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
    #app.run(debug=True)
