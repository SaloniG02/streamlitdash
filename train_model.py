import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

X = np.load("X.npy")
Y = np.load("Y.npy")
class_names = np.load("class_names.npy")

num_classes = len(class_names)
print("Training with", num_classes, "classes")

X = X / 255.0
X = np.stack([X, X, X], axis=-1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    MaxPool2D(2,2),

    Conv2D(64, (3,3), activation="relu"),
    MaxPool2D(2,2),

    Conv2D(128, (3,3), activation="relu"),
    MaxPool2D(2,2),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

model.save("models/sound_model.h5")
print("Model saved!")
