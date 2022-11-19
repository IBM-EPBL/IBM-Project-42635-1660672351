import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score,confusion_matrix
import Inputscript

import requests

# NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
API_KEY = "Qr8Nnabo_P7ke1jW3Y_ZnJOll1Sc0NF_retFqgKGdW_E"
token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
 API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
mltoken = token_response.json()["access_token"]

header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

# NOTE: manually define and pass the array(s) of values to be scored in the next line
payload_scoring = {"input_data": [{"fields": ['having_IPhaving_IP_Address','URLURL_Length','Shortining_Service','having_At_Symbol',
             'double_slash_redirecting','Prefix_Suffix','having_Sub_Domain','SSLfinal_State',
              'Domain_registeration_length','Favicon','port','HTTPS_token','Request_URL',
              'URL_of_Anchor','Links_in_tags','SFH','Submitting_to_email','Abnormal_URL',
              'Redirect','on_mouseover','RightClick','popUpWidnow','Iframe',
              'age_of_domain','DNSRecord','web_traffic','Page_Rank','Google_Index',
              'Links_pointing_to_page','Statistical_report'], "values": [1,1,1,1,1,-1,-1,-1,-1,1,1,1,1,-1,-1,1,1,1,0,1,1,1,1,-1,-1,-1,-1,1,0,1]}]}

response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/e133cc12-94ea-4b74-8355-f1174aad062b/predictions?version=2022-11-18', json=payload_scoring,
 headers={'Authorization': 'Bearer ' + mltoken})
print("Scoring response")
print(response_scoring.json())


app = Flask(__name__)


@app.route('/')   #decorator
def phishing_detection():
    return render_template('index.html')


@app.route('/y_predict',methods=['POST'])
def y_predict():
    url = request.form['url']
    checkprediction = Inputscript.main(url)
    prediction = response_scoring.json(checkprediction)
    print(prediction)
    output=prediction[0]
    if(output==-1):
        pred="your website is legitimate. ::"
   
    else:
        pred = "You are in a suspecious site. Be Cautious :("

    

    return render_template('index.html',prediction_text='{}'.format(pred),url=url)

@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    prediction = response_scoring.y_predict([np.array(list(data.values()))])

    output= prediction[0]
    return jsonify(output)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

