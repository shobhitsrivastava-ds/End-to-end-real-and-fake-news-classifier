from flask import Flask,request, url_for, redirect, render_template, jsonify
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.models import load_model
import nltk
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import pandas as pd
import numpy as np
import pickle
import os
app = Flask(__name__)


nltk.download('stopwords')
nltk.download('punkt')
################
maxlen = 700   #
verbose = False#
################

def print_s(obj, verbose=False):
    if verbose:
        if obj=='model':
            print("Loading LSTM W2V Model...")
        elif obj == 'enc':
            print("Loading Encoder...")
        elif obj == 'pre':
            print("Preprocessing the news...")
        elif obj == 'trans':
            print("Transforming the news...")
        elif obj == 'pred':
            print("Predicting Real of Fake")
    else:
        return


def print_f(obj, verbose=False):
    if verbose:
        if obj=='model':
            print("LSTM W2V Model loading complete :)")
        elif obj == 'enc':
            print("Encoder loading complete :)")
        elif obj == 'pre':
            print("Preprocessing complete :)")
        elif obj == 'trans':
            print("Transforming complete :)")
        #elif obj == 'pred':
            #print("Predicting Real of Fake")
    else:
        return


def loadModels(model_path, encoder_path, verbose=False):
    model_path = os.path.join(model_path, "model_v1.h5")
    encoder_path = os.path.join(encoder_path, "tokenizer.h5")
    print_s('model', verbose)
    model = load_model(model_path)
    print_f('model', verbose)
    print_s('enc', verbose)
    with open(encoder_path, 'rb') as pickle_file:
        encoder = pickle.load(pickle_file)
        #print(type(model),  type(encoder))
    print_f('enc', verbose)
    return model, encoder

def preprocess(par, verbose=False):
    print_s('pre', verbose)
    X = []
    stop_words = set(nltk.corpus.stopwords.words("english"))
    tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
    tmp = []
    sentences = nltk.sent_tokenize(par)
    for sent in sentences:
        sent = sent.lower()
        tokens = tokenizer.tokenize(sent)
        filtered_words = [w.strip() for w in tokens if w not in stop_words and len(w) > 1]
        tmp.extend(filtered_words)
        #X.append(tmp)
    print_f('pre', verbose)
    return tmp


def transform(X, maxlen, verbose=False):
    print_s('trans', verbose)
    #X = preprocess(txt)
    tmp = np.array(X)
    tmp = tmp.reshape(1, tmp.shape[0])
    X = encoder.texts_to_sequences(tmp.tolist())
    print_f('trans', verbose)
    return pad_sequences(X, maxlen)


def predict_news(txt, maxlen, clf_model, txt_encoder, verbose=False):
    X = preprocess(txt, verbose)
    X = transform(X, maxlen, verbose)
    print_s(verbose, 'pred')
    y = clf_model.predict(X)
    if y>0.5:
        return "Real"
    else:
        return "Fake"

model, encoder = loadModels('models', 'models', verbose=verbose)



#model = load_model('deployment_28042020')

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict',methods=['POST'])
def predict():
    #int_features = [x for x in request.form.values()]
    #final = np.array(int_features)
    #data_unseen = pd.DataFrame([final], columns = cols)
    #prediction = predict_model(model, data=data_unseen, round = 0)
    #prediction = int(prediction.Label[0])
    form_txt = str(request.form)
    y = predict_news(form_txt, maxlen, model, encoder, verbose=verbose)

    return render_template('home.html',pred='News is {}'.format(y))

'''
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.Label[0]
    return jsonify(output)

'''

if __name__ == '__main__':
    app.run(debug=True)