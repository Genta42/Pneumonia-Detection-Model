import numpy as np
import os
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2

# Load images function with error handling and proper label parsing
def load_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            if "NORMAL" in filename:  # Normal lungs
                labels.append(0)
            elif "BACTERIA" in filename:  # Bacterial pneumonia
                labels.append(1)
            elif "VIRUS" in filename:  # Viral pneumonia
                labels.append(2)
            else:
                print(f"Skipping file {filename}: label not recognized")
        else:
            print(f"Failed to load image {filename}")
    return np.array(images), np.array(labels)

# Paths to your datasets
normal_dir = r'/Users/genta/Desktop/Genta/Programming/Pneumonia AI/chest_xray/train/NORMAL'
pneumonia_dir = r'/Users/genta/Desktop/Genta/Programming/Pneumonia AI/chest_xray/train/PNEUMONIA'

# Load images
normal_images, normal_labels = load_images(normal_dir)
pneumonia_images, pneumonia_labels = load_images(pneumonia_dir)

# Combine datasets
images = np.concatenate((normal_images, pneumonia_images), axis=0)
labels = np.concatenate((normal_labels, pneumonia_labels), axis=0)

# Shuffle and split data
indices = np.arange(images.shape[0])
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Calculate class weights to handle class imbalance
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights[i] for i in range(len(class_weights))}

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.2,  
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

# Custom generator to apply class weights
def custom_generator(generator, class_weights):
    while True:
        x_batch, y_batch = next(generator)
        # Adjust sample weights based on the class
        sample_weights = np.array([class_weights[int(label)] for label in y_batch])
        yield x_batch, y_batch, sample_weights

# Transfer Learning with VGG16
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Freezing the layers of VGG16
for layer in base_model.layers:
    layer.trainable = False

# Adding custom layers on top of VGG16
x = base_model.output
x = Flatten()(x)
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
predictions = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model with Adam optimizer and a custom learning rate
optimizer = Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks for early stopping and learning rate reduction
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001)
model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True, monitor='val_loss')

# Training the model with the custom generator (Note: Removed `class_weight` from `fit()`)
history = model.fit(
    custom_generator(datagen.flow(X_train, y_train, batch_size=32), class_weights),
    validation_data=(X_test, y_test),
    epochs=50,
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    steps_per_epoch=len(X_train) // 32
)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the final model
model.save('final_model.keras')