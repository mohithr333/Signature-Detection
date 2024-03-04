from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from skimage import measure, morphology
from skimage.color import label2rgb
import matplotlib.pyplot as plt

UPLOAD_FOLDER = 'static/uploads/'
SIGNATURE_FOLDER = 'static/signatures/'
CROPPED_SIGNATURE_FOLDER = 'static/cropped_signatures'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SIGNATURE_FOLDER'] = SIGNATURE_FOLDER
app.config['CROPPED_SIGNATURE_FOLDER'] = CROPPED_SIGNATURE_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = load_model('signature_model.h5')

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Constants for signature detection
CONSTANT_PARAMETER_1 = 84
CONSTANT_PARAMETER_2 = 250
CONSTANT_PARAMETER_3 = 100
CONSTANT_PARAMETER_4 = 18


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_signature(image_path):
    img = cv2.imread(image_path, 0)
    img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]

    blobs = img > img.mean()
    blobs_labels = measure.label(blobs, background=1)

    the_biggest_component = 0
    total_area = 0
    counter = 0
    signature_boxes = []  # Store bounding boxes of detected signatures

    for region in measure.regionprops(blobs_labels):
        if region.area > 10:
            total_area += region.area
            counter += 1
        if region.area >= 250:
            if region.area > the_biggest_component:
                the_biggest_component = region.area

            minr, minc, maxr, maxc = region.bbox
            signature_boxes.append((minc, minr, maxc - minc, maxr - minr))  # (x, y, w, h)

    average = total_area / counter

    a4_small_size_outlier_constant = ((average / CONSTANT_PARAMETER_1) * CONSTANT_PARAMETER_2) + CONSTANT_PARAMETER_3
    a4_big_size_outlier_constant = a4_small_size_outlier_constant * CONSTANT_PARAMETER_4

    pre_version = morphology.remove_small_objects(blobs_labels, a4_small_size_outlier_constant)
    component_sizes = np.bincount(pre_version.ravel())
    too_small = component_sizes > a4_big_size_outlier_constant
    too_small_mask = too_small[pre_version]
    pre_version[too_small_mask] = 0

    pre_version_path = 'static/signatures/pre_version.png'
    plt.imsave(pre_version_path, pre_version)

    img = cv2.imread(pre_version_path, 0)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    output_path = 'static/signatures/output.png'
    cv2.imwrite(output_path, img)

    return output_path, signature_boxes

def crop_signatures(image_path, output_image_path, signature_boxes):
    # Load the original image
    output_image = cv2.imread(output_image_path)

    # Iterate over each bounding box
    cropped_signature_filenames = []
    for i, box in enumerate(signature_boxes):
        x, y, w, h = box
        # Crop the signature from the original image
        signature = output_image[y:y + h, x:x + w]

        cropped_signature_filename = f'cropped_signature-{i}.png'
        # Save the cropped signature
        cropped_signature_path = os.path.join(app.config['CROPPED_SIGNATURE_FOLDER'], cropped_signature_filename)
        cv2.imwrite(cropped_signature_path, signature)
        cropped_signature_filenames.append(cropped_signature_filename)

        print("Cropped signature filenames:", cropped_signature_filenames)  # Add this line to print the filenames
        print("Cropped signature paths:", cropped_signature_path)  # Add this line to print the paths

    return cropped_signature_filenames

@app.route('/')
def upload_form():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return render_template('index.html')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html')

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        flash('Image successfully uploaded')
        return render_template('index.html', filename=filename)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/detect_signature', methods=['POST'])
def detect_signature_route():
    if 'filename' not in request.form:
        return render_template('index.html')

    filename = request.form['filename']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    signature_path = detect_signature(image_path)

    return render_template('index.html', filename=filename, signature_path=signature_path)

@app.route('/crop_signatures', methods=['POST'])
def crop_signature_route():
    if 'filename' not in request.form:
        return render_template('index.html')

    filename = request.form['filename']
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    output_image_path, signature_boxes = detect_signature(image_path)
    cropped_signature_filenames = crop_signatures(image_path, output_image_path, signature_boxes)

    cropped_signature_paths = [os.path.join(app.config['CROPPED_SIGNATURE_FOLDER'], filename) for filename in
                               cropped_signature_filenames]

    return render_template('index.html', filename=filename, signature_path=output_image_path,
                           cropped_signature_paths=cropped_signature_paths)

@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display_signature/<filename>')
def display_signature(filename):
    return redirect(url_for('static', filename='signatures/' + filename), code=301)

@app.route('/display_cropped_signature/<filename>')
def display_cropped_signature(filename):
    return redirect(url_for('static', filename='cropped_signatures/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
