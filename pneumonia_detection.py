import numpy as np
import os
import cv2
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.regularizers import l2

# Load images function with RGB conversion
def load_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        img = cv2.imread(os.path.join(directory, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, (128, 128))  # Resize to 128x128 pixels
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # Convert to RGB
            img = img / 255.0  # Normalize pixel values
            images.append(img)
            if "IM" in filename:  # Assuming "IM" in filename indicates normal lungs
                labels.append(0)
            elif "bacteria" in filename:  # Bacterial pneumonia
                labels.append(1)
            elif "virus" in filename:  # Viral pneumonia
                labels.append(2)
    return np.array(images), np.array(labels)

# Paths to your datasets
normal_dir = r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\ChestXRay2017\chest_xray\test\NORMAL'
pneumonia_dir = r'D:\Downloads\AI Pneumonia Detector from X-Ray Images\ChestXRay2017\chest_xray\test\PNEUMONIA'

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

# Training the model
history = model.fit(datagen.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=50,
                    class_weight=class_weights,
                    callbacks=[early_stopping, reduce_lr, model_checkpoint])

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Save the final model
model.save('final_model.keras')
