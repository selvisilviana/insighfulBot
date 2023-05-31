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

 [Routing untuk Halaman Utama atau Home]	
@app.route("/")
def beranda():
    return render_template('index.html')

# [Routing untuk API]
@app.route("/api/deteksi", methods=['POST'])
def api_deteksi():
    if request.method == 'POST':
        # Ambil input dari pengguna
        input_text = request.form['input_text']
        
        # Lakukan preprocessing pada input_text jika diperlukan
        
        # Lakukan prediksi menggunakan model LSTM
        prediksi = lstm_predict(input_text)
        
        # Return hasil prediksi dengan format JSON
        return jsonify({"prediksi": prediksi})

# Prediksi menggunakan model LSTM
def lstm_predict(input_text):
    # Lakukan transformasi pada input_text jika diperlukan
    
    # Lakukan prediksi menggunakan model LSTM
    # Contoh:
    # input_data = transform_input_text(input_text)
    # prediksi = model.predict(input_data)
    
    # Lakukan post-processing pada hasil prediksi jika diperlukan
    
    return prediksi
    
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
   	
	cv = CountVectorizer(vocabulary=pickle.load(open('kbest_feature.pickle', 'rb')))
	
	# Load model phishing yang telah ditraining
	model = load('chatbot_model.h5')

	# Run Flask di Google Colab menggunakan ngrok
	run_with_ngrok(app)
	app.run()