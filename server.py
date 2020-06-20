import os
from flask import Flask, flash, render_template, redirect, request, url_for, send_file
from numpy import pi, squeeze
from keras.models import load_model
from PIL import Image
import numpy
from utils import generate_random_name, is_valid_file_type, load_image, classes

app = Flask(__name__)
app.config['SECRET_KEY'] = "yashandsal"
model = load_model('neural_net.h5')


@app.route('/images/<filename>', methods=['GET'])
def images(filename):
    return send_file(os.path.join('/tmp/images/', filename))

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
        return render_template('home.html')

    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No file was uploaded.')
            return redirect(request.url)

        image_file = request.files['image']

        if image_file.filename == '':
            flash('No image was uploaded.')
            return redirect(request.url)

        if image_file and is_valid_file_type(image_file.filename):
            passed = False
            try:
                filename = generate_random_name(image_file.filename)
                filepath = os.path.join('/tmp/images/', filename)
                image_file.save(filepath)
                passed = True
            except Exception:
                passed = False

            if passed:
                return redirect(url_for('predict', filename=filename))
            else:
                flash('An error occurred, try again.')
                return redirect(request.url)

@app.route('/predict/<filename>', methods=['GET'])
def predict(filename):
    image = Image.open(os.path.join('/tmp/images/', filename))
    image = image.resize((30,30))
    image = numpy.expand_dims(image, axis=0)
    image = numpy.array(image)
    print(image.shape)
    pred = model.predict_classes([image])[0]
    prediction = classes[pred+1]
    """ image_url = url_for('images', filename=filename)
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image_data = load_image(image_path)
    predictions = model.predict_proba(image_data) """
    return render_template('predict.html', prediction=prediction)


@app.errorhandler(500)
def server_error(error):
    return render_template('error.html'), 500


