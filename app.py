from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('HDClassifier.pkl', 'rb'))
le = pickle.load(open('le.pkl', 'rb'))
ct = pickle.load(open('ct.pkl', 'rb'))
ct1 = pickle.load(open('ct1.pkl', 'rb'))
ct3 = pickle.load(open('ct3.pkl', 'rb'))
sc1 = pickle.load(open('sc1.pkl', 'rb'))
sc2 = pickle.load(open('sc2.pkl', 'rb'))
sc3 = pickle.load(open('sc3.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')

@app.route('/prediction', methods =['POST'])
def home():

    
    age = int(request.form['age'])
    gender = int(request.form['gender'])
    cp_type = request.form['CP_type']
    bp = int(request.form['bp'])
    cholestoral = int(request.form['cholestoral'])
    fasting = int(request.form['fasting'])
    ecg = request.form['ecg']
    max_hr = int(request.form['max_hr'])
    exercise = int(request.form['exercise'])
    old_peak = float(request.form['old_peak'])
    slope = request.form['slope']
    
    arr = [[age,gender,cp_type,bp,cholestoral,fasting,ecg,max_hr,exercise,old_peak,slope]]
    print(arr)
    # arr[0][1] =(le.transform([[arr[0][1]]]))[0]
    arr = ct.transform(arr)
    arr = ct1.transform(arr)
    arr = ct3.transform(arr)
    # # arr[16] = le.transform(arr[16])

    arr[0][10:11] = sc1.transform([arr[0][10:11]])
    arr[0][12:14] = sc2.transform([arr[0][12:14]])[0]
    arr[0][15:16] = sc3.transform([arr[0][15:16]])



    pred = model.predict(arr)
    proba = round(model.predict_proba(arr)[0][0]*100,2)
    return render_template('after.html', pred =pred ,proba = proba, arr=arr)
    

if __name__ == "__main__":
    app.run(debug=True)