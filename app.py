from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Ensure this directory exists
app.config['SECRET_KEY'] = 'your_secure_secret_key_here'  # Set a secure secret key
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Load the trained model (ensure the path is accurate)
model = load_model(r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\my_flask_app\final_model.keras')

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    """Read an image from the file, resize, convert to RGB, and normalize it for model prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Read in color
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))  # Resize to the input size expected by the model
    img = img.astype(np.float32) / 255.0  # Convert to float32 and normalize
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

def get_prediction_label(prediction_index):
    """Map numeric prediction indices to meaningful labels."""
    labels = {0: "Normal", 1: "Bacteria", 2: "Virus"}
    return labels.get(prediction_index, "Unknown")  # Return "Unknown" if index is not found

@app.route('/')
def home():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload_and_predict', methods=['POST'])
def upload_and_predict():
    """Handle file upload and predict the image class."""
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        img = preprocess_image(file_path)
        if img is not None:
            prediction = model.predict(img)
            predicted_class_index = np.argmax(prediction, axis=1)[0]
            predicted_label = get_prediction_label(predicted_class_index)
            confidence = np.max(prediction) * 100  # Multiply by 100 to convert to percentage
            result = {
                'filename': filename,
                'prediction': predicted_label,
                'confidence': f"{confidence:.2f}%"  # Format confidence to two decimal places
            }
            return render_template('result.html', result=result)
        else:
            flash('Could not process the image, please upload another.')
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
