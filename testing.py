import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# Paths to the directories containing normal and pneumonia images
NORMAL_DIR = r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\ChestXRay2017\chest_xray\test\NORMAL'
PNEUMONIA_DIR = r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\ChestXRay2017\chest_xray\test\PNEUMONIA'

# Load the pre-trained Keras model
model = load_model(r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\Medical Image Analysis\final_model.keras')

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error processing {image_path}: Image not found.")
        return None
    img = cv2.resize(img, (128, 128))
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_image(image_path):
    """Predict the class of an image using the pre-trained model."""
    img = preprocess_image(image_path)
    if img is not None:
        prediction = model.predict(img)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        return predicted_class_index
    else:
        return None

def batch_process(directory):
    """Process a batch of images and log the results."""
    results = []
    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)  # Full path to the image
            predicted_class_index = predict_image(image_path)
            
            # Determine actual label based on filename
            if "IM" in filename:
                actual_label = 'Normal'
            elif "bacteria" in filename:
                actual_label = 'Bacteria'
            elif "virus" in filename:
                actual_label = 'Virus'
            else:
                actual_label = 'Unknown'
                
            if predicted_class_index is not None:
                # Adjust the logic here based on how your model is trained to predict:
                if predicted_class_index == 0:
                    predicted_label = 'Normal'
                elif predicted_class_index == 1:
                    predicted_label = 'Bacteria'
                elif predicted_class_index == 2:
                    predicted_label = 'Virus'
                else:
                    predicted_label = 'Unknown'
                    
                results.append({
                    'filename': filename,
                    'actual_label': actual_label,
                    'predicted_label': predicted_label
                })
            else:
                print(f"Error processing {filename}: Image could not be processed.")
    return results

def get_unique_filename(base_filename):
    """Generate a unique filename if the base filename already exists."""
    if not os.path.exists(base_filename):
        return base_filename
    
    filename, extension = os.path.splitext(base_filename)
    counter = 1
    new_filename = f"{filename}_{counter}{extension}"
    
    while os.path.exists(new_filename):
        counter += 1
        new_filename = f"{filename}_{counter}{extension}"
    
    return new_filename

# Main execution block
if __name__ == "__main__":
    # Process normal images
    normal_results = batch_process(NORMAL_DIR)
    
    # Process pneumonia images
    pneumonia_results = batch_process(PNEUMONIA_DIR)
    
    # Combine the results from both directories
    results = normal_results + pneumonia_results
    
    # Determine a unique filename for the CSV file
    base_filename = 'prediction_results.csv'
    unique_filename = get_unique_filename(base_filename)
    
    # Save the results to the unique filename
    results_df = pd.DataFrame(results)
    results_df.to_csv(unique_filename, index=False)
    print(f"Results have been saved to '{unique_filename}'.")
