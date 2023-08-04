import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load the FER2013 dataset or your custom dataset
# Replace 'path_to_csv_file' with the path to your CSV file containing the dataset
data = pd.read_csv('/content/drive/MyDrive/train.csv')

# Process the image data
def preprocess_image(image_data):
    pixels = image_data.split()
    image = np.array(pixels, dtype='uint8').reshape((48, 48, 1))
    return image

data['pixels'] = data['pixels'].apply(preprocess_image)
X = np.stack(data['pixels'].values)
y = data['emotion'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation for training set
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Data augmentation for validation set (only rescaling)
val_datagen = ImageDataGenerator(rescale=1./255)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
batch_size = 32
epochs = 20

train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
val_generator = val_datagen.flow(X_test, y_test, batch_size=batch_size)

history = model.fit_generator(train_generator, steps_per_epoch=len(X_train) // batch_size,
                              validation_data=val_generator, validation_steps=len(X_test) // batch_size,
                              epochs=epochs)

# Save the trained model to a file
model.save('facial_expression_model.h5')

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy:', test_accuracy)
