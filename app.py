# =[Modules dan Packages]========================

from flask import Flask,render_template,request,jsonify
from flask_ngrok import run_with_ngrok
import pandas as pd
import numpy as np
import re
import pickle
import nltk
import joblib
import pickle
from joblib import load

#import package untuk phishing detection
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

#import package untuk safebot
from process import preparation, generate_response

# download nltk
preparation()

# =[Variabel Global]=============================

app   = Flask(__name__, static_url_path='/static')
model = None

# Text Pre-Processing function untuk API phishing detection
def text_preprocessing_process(text):
    tokens = tokenizer.tokenize(text)
    tokens_stemmed = [stemmer.stem(token) for token in tokens]
    processed_text = ' '.join(tokens_stemmed)
    return processed_text

# =[Routing]=====================================

# [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk Halaman Utama atau Home]	
@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

# Routing for API phishing
@app.route("/api/deteksi", methods=['POST'])
def apiDeteksi():
    prediction_input = ""
    
    if request.method == 'POST':
        prediction_input = request.form['data']
        
        # Text Pre-Processing
        prediction_input = text_preprocessing_process(prediction_input)
        
        # Vectorization
        feature = cv.transform([prediction_input])
        
        # Prediction (Web Phishing or Web Aman)
        prediction = model.predict(feature)
        
        if prediction == 0:
            hasil_prediksi = "Web Phishing"
        else:
            hasil_prediksi = "Web Aman"
        
        # Return the prediction result in JSON format
        return jsonify({
            "data": hasil_prediksi,
        })
    
# Routing for API response chatbot
@app.route("/get")
def get_bot_response():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

# =[Main]========================================

if __name__ == '__main__':
        
    #Setup Phishing
	tokenizer = RegexpTokenizer(r'[A-Za-z]+')
	stemmer = SnowballStemmer("english")
   	
	cv = CountVectorizer(vocabulary=pickle.load(open('feature.pickle', 'rb')))
	
	# Load model phishing yang telah ditraining
	model = load('model_phishing_lr.model')

	# Run Flask di Google Colab menggunakan ngrok
	run_with_ngrok(app)
	app.run()