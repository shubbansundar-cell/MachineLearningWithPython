import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# -----------------------------------------------------
# LOAD DATA (already loaded in FCC notebook)
# -----------------------------------------------------
# dataset = pd.read_csv("insurance.csv")

# -----------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------

# Convert categorical columns to numeric
dataset = pd.get_dummies(dataset, columns=['sex', 'smoker', 'region'])

# Split features and labels
labels = dataset.pop('expenses')

# Train-test split (80/20)
train_dataset, test_dataset, train_labels, test_labels = train_test_split(
    dataset, labels, test_size=0.2, random_state=42
)

# -----------------------------------------------------
# NORMALIZATION
# -----------------------------------------------------

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_dataset))

# -----------------------------------------------------
# MODEL
# -----------------------------------------------------

model = keras.Sequential([
    normalizer,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error']
)

# -----------------------------------------------------
# TRAINING
# -----------------------------------------------------

history = model.fit(
    train_dataset,
    train_labels,
    validation_split=0.2,
    epochs=100,
    verbose=0
)

# -----------------------------------------------------
# EVALUATION (FCC TEST)
# -----------------------------------------------------

loss, mae = model.evaluate(test_dataset, test_labels, verbose=0)
print("Mean Absolute Error:", mae)

# -----------------------------------------------------
# PREDICTIONS (FINAL CELL EXPECTS THIS)
# -----------------------------------------------------

test_predictions = model.predict(test_dataset).flatten()

# Return required variables for FCC testing
mae, test_predictions
