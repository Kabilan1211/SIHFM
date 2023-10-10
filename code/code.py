import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load non-image features from Excel
excel_file = "data.xlsx"
df = pd.read_excel(excel_file)

# Extract non-image features
moisture = df["Moisture"].values
temperature = df["Temperature"].values
humidity = df["Humidity"].values
time_taken = df["TimeTaken"].values

# Normalize numerical features
scaler = StandardScaler()
moisture = scaler.fit_transform(moisture.reshape(-1, 1))
temperature = scaler.fit_transform(temperature.reshape(-1, 1))
humidity = scaler.fit_transform(humidity.reshape(-1, 1))
time_taken = scaler.fit_transform(time_taken.reshape(-1, 1))

# Load and preprocess your images from a folder
image_folder = "images"  # Replace with the path to your image folder
image_data = []
labels = []

for filename in os.listdir(image_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (150, 150))  # Resize images as needed
        img = img / 255.0  # Normalize pixel values
        image_data.append(img)
        # Assuming you have a way to determine labels for each image, add it here
        # labels.append(label)

# Convert lists to NumPy arrays
image_data = np.array(image_data)
# labels = np.array(labels)  # Uncomment and replace with your labels

# Split your data into training and testing sets
# Replace 'y_train' and 'y_test' with your actual labels
X_train_img, X_test_img, y_train, y_test = train_test_split(image_data, labels, test_size=0.2, random_state=42)
X_train_num = np.column_stack((moisture, temperature, humidity, time_taken))
X_test_num = np.column_stack((moisture, temperature, humidity, time_taken))

# Define the CNN architecture for image processing
image_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
])

# Define a separate neural network for non-image features
numeric_model = models.Sequential([
    layers.Input(shape=(4,)),  # Adjust the input shape based on the number of features
    layers.Dense(64, activation='relu'),
])

# Combine the two models (image and numeric) using a Concatenate layer
combined_input = layers.concatenate([image_model.output, numeric_model.output])

# Add additional layers for classification
x = layers.Dense(64, activation='relu')(combined_input)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation='sigmoid')(x)  # Binary classification, adjust as needed

# Create the final model
final_model = models.Model(inputs=[image_model.input, numeric_model.input], outputs=output)

# Compile the model
final_model.compile(optimizer='adam',
                    loss='binary_crossentropy',  # Change for your task
                    metrics=['accuracy'])

# Data augmentation and preprocessing for images
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Train the model with both image and non-image features
history = final_model.fit(
    [X_train_img, X_train_num],
    y_train,
    epochs=10,  # Adjust the number of epochs
    validation_data=([X_test_img, X_test_num], y_test)
)

# Save the trained model
final_model.save('multimodal_classification_model.h5')
