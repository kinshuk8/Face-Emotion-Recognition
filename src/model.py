import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras import layers

def create_model():
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(48, 48, 3))  # Use 3 channels (RGB)

    base_model.trainable = False  # Freeze the base model layers

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')  # 7 emotions
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
