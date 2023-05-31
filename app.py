import json
import random
from flask import Flask, render_template, request, jsonify
from flask_ngrok import run_with_ngrok

app = Flask(__name__, static_url_path='/static')
run_with_ngrok(app)

# Routing untuk Halaman Utama atau Home
@app.route("/")
def beranda():
    return render_template('index.html')

# Routing untuk Halaman Chatbot
@app.route("/chatbot")
def chatbot():
    return render_template('chatbot.html')

# Routing untuk API Deteksi
@app.route("/api/deteksi", methods=['POST'])
def api_deteksi():
    if request.method == 'POST':
        # Lakukan pengolahan data dan prediksi
        # ...
        return jsonify({"prediksi": prediksi})

# Fungsi untuk menghasilkan respons berdasarkan input pengguna
def generate_response(user_input):
    for i in range(len(patterns)):
        if user_input in patterns[i]:
            return random.choice(responses[i])
    return "Maaf, saya tidak mengerti."

# Routing untuk mendapatkan respons chatbot
@app.route("/get")
def get_bot_responses():
    user_input = str(request.args.get('msg'))
    result = generate_response(user_input)
    return result

# Membaca dataset dari file JSON
with open('static/dataset.json') as file:
    dataset = json.load(file)

# Mendapatkan tag, pola, dan respons dari dataset
tag = dataset['tag']
patterns = dataset['patterns']
responses = dataset['responses']

# Fungsi-fungsi lainnya di dalam aplikasi Flask
# ...

if __name__ == '__main__':
    app.run()
