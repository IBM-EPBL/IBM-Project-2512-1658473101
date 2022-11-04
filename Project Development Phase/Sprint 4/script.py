import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, make_response
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from predict_cvd import PredictCVD

CVDs = [
    'Left Bundle Branch Block',
    'Normal',
    'Premature Atrial Contraction',
    'Premature Ventricular Contractions',
    'Right Bundle Branch Block',
    'Ventricular Fibrillation'
]

app = Flask(__name__)


@app.route("/")
@app.route("/index.html")
def home():
    return render_template("index.html")


@app.route("/info.html")
def info():
    return render_template("info.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict_cvd():
    if request.method == 'POST':
        print(request.files)
        request.files['image'].save("ecg.png")
        predictor = PredictCVD()
        response = make_response("response")
        response.data = "Hello"
        predictor.predict("ecg.png")
        return response
    return "Method not allowed"


@app.route("/predict.html", methods=['GET', 'POST'])
def predict():
    return render_template("predict.html")


@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')


if __name__ == "__main__":
    app.run(debug=True)
