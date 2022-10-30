from flask import Flask, request
from flask.templating import render_template
import numpy as np
import pandas as pd 
import model
import pickle

predictor = pickle.load(open('pickledModel.sav', 'rb'))

app = Flask(__name__, template_folder='templates')

@app.route("/", methods=['GET','POST'])
def bcc():
    prediction = None
    if request.method == "POST":
        CT = request.form["CT"]
        uSize = request.form["uSize"]
        uShape = request.form["uShape"]
        mA = request.form["mA"]
        SECS = request.form["SECS"]
        BN = request.form["BN"]
        BC = request.form["BC"]
        NN = request.form["NN"]
        Mit = request.form["Mit"]
        pred_data = np.array([CT, uSize, uShape, mA, SECS, BN, BC,NN, Mit])
        pred_data = pred_data.reshape(1,9)
        col = model.cols
        df = pd.DataFrame(data=pred_data)
        prediction = predictor.predict(df)
        if prediction[0] == 2:
            pred = "Benign"
        elif prediction[0] == 4:
            pred = "Malignant"
        return render_template("base.html", pred = pred)

    else:
        return render_template("base.html")
        

if __name__ == "__main__":
    app.run(debug=True)
