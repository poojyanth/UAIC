import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cProfile

def load_and_process_mnist():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension for grayscale images
    X_test = np.expand_dims(X_test, axis=-1)

    X_train = X_train.astype("float32") / 255.0  # Normalize pixel values
    X_test = X_test.astype("float32") / 255.0

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_and_process_mnist()

batch_size = 32

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(X_train)).batch(batch_size)

# Define your model
def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = X_train.shape[1:]
num_classes = 10  # MNIST has 10 classes

def main():
    # Your deep learning code here
    model = create_model(input_shape, num_classes)

    model.compile(optimizer=Adam(), loss=SparseCategoricalCrossentropy(), metrics=['accuracy'])

    epochs = 2
    model.fit(train_dataset, epochs=epochs)

    # Evaluate the model
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)
    accuracy = model.evaluate(test_dataset)[1]
    print(f"Accuracy on the test set: {accuracy}")

if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()

    main()

    profiler.disable()

    # Save profiling data to a .prof file
    profiler.dump_stats("profile_results.prof")
