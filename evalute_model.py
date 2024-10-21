import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import validation_generator

# Load the model
model_file = 'model_1.h5'
print(f"Evaluating model: {model_file}")
model = load_model(model_file)

# Evaluate on validation data
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

