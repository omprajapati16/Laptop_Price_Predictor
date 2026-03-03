from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load model
pipe = pickle.load(open("pipe.pkl", "rb"))

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Company": request.form["Company"],
            "TypeName": request.form["TypeName"],
            "Inches": float(request.form["Inches"]),
            "Cpu": request.form["Cpu"],
            "Ram": int(request.form["Ram"].replace("GB","")),
            "Memory": request.form["Memory"],
            "Gpu": request.form["Gpu"],
            "Weight": float(request.form["Weight"])
        }

        df = pd.DataFrame([data])

        df = pd.get_dummies(df)

        # Load training columns
        model_columns = pickle.load(open("model_columns.pkl", "rb"))

        # Match training columns
        df = df.reindex(columns=model_columns, fill_value=0)

        prediction = np.exp(pipe.predict(df)[0])

        return jsonify({"price": round(prediction, 2)})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)