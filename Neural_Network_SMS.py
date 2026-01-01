import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

# -----------------------------------------------------
# DATA (already loaded by FCC notebook)
# -----------------------------------------------------
# train_data, test_data are already provided
# train_data = (train_sentences, train_labels)
# test_data = (test_sentences, test_labels)

train_sentences, train_labels = train_data
test_sentences, test_labels = test_data

# -----------------------------------------------------
# TEXT VECTORIZATION
# -----------------------------------------------------

max_vocab_size = 10000
sequence_length = 200

vectorize_layer = layers.TextVectorization(
    max_tokens=max_vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length
)

vectorize_layer.adapt(train_sentences)

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------

model = keras.Sequential([
    vectorize_layer,
    layers.Embedding(max_vocab_size, 64),
    layers.GlobalAveragePooling1D(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -----------------------------------------------------
# TRAINING
# -----------------------------------------------------

model.fit(
    train_sentences,
    train_labels,
    epochs=10,
    validation_data=(test_sentences, test_labels),
    verbose=0
)

# -----------------------------------------------------
# REQUIRED FUNCTION
# -----------------------------------------------------

def predict_message(message):
    prediction = model.predict([message])[0][0]
    label = "spam" if prediction >= 0.5 else "ham"
    return [float(prediction), label]
