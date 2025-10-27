import pickle 
from flask import Flask, render_template, request , app, jsonify, url_for
import numpy as np
import pandas as pd 



app = Flask(__name__)


# Loaded the model (  This is the heart of the Project  )
regmodel = pickle.load(open("models/reg_model.pkl","rb"))  
scaling = pickle.load(open("models/scaler.pkl","rb"))

@app.route('/')
def home():
    return render_template('home.html')


@app.route('/predict_api',methods = ["POST"])   # this is an API 
def predict_api():

    data = request.json['data']  # it basically store the data gets inputed in the Form and store it in the form of Json 
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))
    new_data  = scaling.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)[0]
    print(output[0])  # this prints the output in the terminal 

    
    return jsonify(output)  # this will take the output and send it to the server 


@app.route('/predict',methods = ["POST"]) 

def predict():
    data = [float(x) for x in request.form.values()]
    final_input = scaling.transform(np.array(data).reshape(1,-1))
    print(final_input)

    # Prediction 
    output = regmodel.predict(final_input)[0]
    


    dataset = pd.read_csv("house_prices_dataset.csv")
    avg_price = np.mean(dataset["price"])


    avg_price = int(np.mean(dataset["price"]))
    # Fix negatives or zeros with average
    if output <= 0:
        output = int(avg_price)

    return render_template("home.html",prediction_text = "The house price prediction is {}".format(output))

     






# this indicates the this app.py file is runned directly from hete , not by importing 
if __name__ == "__main__":
    app.run(debug=True)




