import os
import urllib
import keras
import numpy as np
import training
import nail_model
import json
from keras.preprocessing import image
from flask import Flask, request, redirect, flash, url_for, Session
from werkzeug.utils import secure_filename
from keras.applications.vgg16 import preprocess_input


UPLOAD_FOLDER = '/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.mkdir(UPLOAD_FOLDER)
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
DECISION_THRESHOLD = 0.5

sess = Session()
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SESSION_TYPE'] = 'filesystem'
app.secret_key = '07145d5b193b4d3b909f218bccd65be7'

model, graph = nail_model.get_model((224, 224, 3), weights=None)
model.load_weights(os.path.join(training.MODEL_FINAL_SAVE_DIR, training.MODEL_FILENAME))

def predict(file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    global graph
    with graph.as_default():
        result = model.predict(img)
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