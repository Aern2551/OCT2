from flask import Flask, request, jsonify, send_from_directory
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
import logging
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.getLogger('tensorflow').setLevel(logging.ERROR)

app = Flask(__name__)

# Load the AI model
MODEL_PATH = r"D:\OCT\retinal-oct-images-classification\code\old\testimg\baseline_model.keras"
model = tf.keras.models.load_model(MODEL_PATH)

# Directory to save uploaded images
UPLOAD_FOLDER = 'uploaded_images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class names for mapping predictions
CLASS_NAMES = ["CNV", "DME", "DRUSEN", "NORMAL"]

# Serve uploaded images
@app.route('/uploaded_images/<filename>')
def serve_uploaded_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/')
def home():
    return send_from_directory(r"D:\OCT\retinal-oct-images-classification\wad\Frontend", 'index.html')

@app.route('/reference_page')
def reference_page():
    return send_from_directory(r"D:\OCT\retinal-oct-images-classification\wad\Frontend", 'reference_page.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Predict using the loaded model
    try:
        image = load_img(filepath, target_size=(224, 224))
        image = img_to_array(image) / 255.0
        image = np.expand_dims(image, axis=0)

        predictions = model.predict(image)

        # Map predictions to class names
        class_confidences = {CLASS_NAMES[i]: round(float(conf) * 100, 2) for i, conf in enumerate(predictions[0])}

    except Exception as e:
        print(f"Error during upload processing: {e}")
        return jsonify({'error': str(e)}), 500

    return jsonify({
        'message': 'File uploaded successfully',
        'class_confidences': class_confidences,
        'filepath': f'/uploaded_images/{filename}'
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
