from flask import Flask, request, render_template,url_for
from flask_cors import cross_origin
from werkzeug.utils import secure_filename
import sqlite3 as sql
import os
from tensorflow.keras.models import model_from_json
import librosa
import pandas as pd
import numpy as np
import pickle


app = Flask(__name__)

@app.route("/")
@cross_origin()
def home():
    return render_template("index.html")

@app.route("/detect", methods = ["GET", "POST"])
@cross_origin()
def detect():
    if request.method == "POST":
        audio = request.files['file']  
        filename = secure_filename(audio.filename)
        audio.save(os.path.join('static/uploads',filename))
        audio_path = os.path.join('static/uploads',filename)

        json_file = open('model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

        loaded_model.load_weights("Emotion_Model.h5")
        print("Loaded model from disk")

        X, sample_rate = librosa.load(audio_path,res_type='kaiser_fast',duration=2.5,sr=44100,offset=0.5)
	                                 
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        newdf = pd.DataFrame(data=mfccs).T

        newdf= np.expand_dims(newdf, axis=2)
        newpred = loaded_model.predict(newdf, 
	                             batch_size=16, 
	                             verbose=1)
        filename = 'labels'
        infile = open(filename,'rb')
        lb = pickle.load(infile)
        infile.close()

        final = newpred.argmax(axis=1)
        final = final.astype(int).flatten()
        final = (lb.inverse_transform((final)))

    return render_template("index.html",prediction=final[0])


if __name__ == "__main__":
    app.run(debug=True)
