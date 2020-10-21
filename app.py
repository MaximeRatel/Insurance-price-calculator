import pickle
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from utils import *

app = Flask(__name__) #Initialisation

# on récupère les modèles
encoder    = pickle.load(open('models/encoder.pkl', 'rb'))
model_cout = pickle.load(open('models/model_cout.pkl', 'rb')) 
model_freq = pickle.load(open('models/model_freq.pkl', 'rb'))

@app.route('/') # Page d'accueil
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
 
    # on récupère les données spécifiées
    param_names = [x for x in request.form]
    param = [x for x in request.form.values()]
    submission = pd.DataFrame(data=[param], columns=param_names)
    
    submission['CSP'] = submission['CSP'].astype('int64', copy=False)
    submission['USAGE'] = submission['USAGE'].astype('int64', copy=False)
    
    submission = OneHotEncoding(submission, encoder)
    
    prediction = model_freq.predict(submission)  * model_cout.predict(submission)

    return render_template('index.html', prediction_text='Votre prime annuelle est de {} euros'.format(np.round(prediction[0], 2)))

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
