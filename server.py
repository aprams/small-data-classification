import os
import urllib
import keras
import numpy as np
import training
import model as mdl
import json
import scipy
import matplotlib.pyplot as plt
from keras.preprocessing import image
from flask import Flask, request, redirect, flash, url_for, Session
from werkzeug.utils import secure_filename
from keras.applications.mobilenetv2 import preprocess_input

# Upload settings
UPLOAD_FOLDER = '/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
DECISION_THRESHOLD = 0.5

# Flask settings
sess = Session()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = '07145d5b193b4d3b909f218bccd65be7'

# Training args
with open(os.path.join(training.MODEL_SAVE_DIR, training.CONFIG_FILE_NAME), 'r') as f:
    model_args = json.load(f)
    image_size = int(model_args['image_size'])

# Model loading
top_model, graph = mdl.get_top_model()
top_model.load_weights(os.path.join(model_args['model_save_dir'], model_args['model_filename']))
with graph.as_default():
    extractor_model = mdl.get_extractor_model(image_size)


def predict(file_path):
    _, file_ext = os.path.splitext(file_path)
    img = scipy.misc.imresize(plt.imread(file_path, format=file_ext), (image_size, image_size))
    img = np.array(img, dtype=np.float32)
    if img.shape[-1] < 3 or len(img.shape) != 3:
        img = np.stack((img,)*3, -1)
    img = np.expand_dims(img, axis=0)
    if img.shape[-1] == 4:
        img = img[:, :, :, :-1]
    img = preprocess_input(img)
    with graph.as_default():
        bottleneck_features = extractor_model.predict(img)
        result = top_model.predict(bottleneck_features)
    return result[0][0]


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['GET', 'POST'])
def predict_endpoint():
    if request.method == 'GET':
        if 'image_url' in request.args:
            image_url = request.args['image_url']
            filename = secure_filename(image_url[image_url.rfind("/")+1:])
            target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            urllib.request.urlretrieve(image_url, target_file_path)
            json_result = {}
            prediction = predict(target_file_path)
            json_result['result'] = 'good' if prediction > DECISION_THRESHOLD else 'bad'
            json_result['sigmoid_output'] = str(prediction)
            return json.dumps(json_result)
    else:
        return ''


@app.route('/upload_predict', methods=['GET', 'POST'])
def upload_file_endpoint():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            target_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(os.path.join(target_file_path))
            json_result = {}
            prediction = predict(target_file_path)
            json_result['result'] = 'good' if prediction > DECISION_THRESHOLD else 'bad'
            json_result['sigmoid_output'] = str(prediction)
            return json.dumps(json_result)
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <p><input type=file name=file>
         <input type=submit value=Upload>
    </form>
    '''


if __name__ == '__main__':
    app.run(host='0.0.0.0')