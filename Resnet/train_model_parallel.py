import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_data
import matplotlib.pyplot as plt
import concurrent.futures
import os
import pickle

def create_model(num_classes):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(epochs, data_dir, batch_size, history_path):
    train_gen, valid_gen = preprocess_data(data_dir, image_size=(256, 256), batch_size=batch_size)
    num_classes = len(train_gen.class_indices)
    
    model = create_model(num_classes)
    history = model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        steps_per_epoch=train_gen.samples // train_gen.batch_size,
        validation_steps=valid_gen.samples // valid_gen.batch_size
    )
    
    # Save the training history
    history_file = os.path.join(history_path, f'history_{epochs}_epochs.pkl')
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)
    
    # Save the model
    model_file = f'resnet50_leaf_disease_model_{epochs}_epochs.h5'
    model.save(model_file)

    return history.history, model_file

def plot_results(history_path, epochs_list):
    plt.figure(figsize=(12, 6))
    
    for epochs in epochs_list:
        history_file = os.path.join(history_path, f'history_{epochs}_epochs.pkl')
        if os.path.exists(history_file):
            with open(history_file, 'rb') as file:
                history = pickle.load(file)
                plt.plot(history['accuracy'], label=f'Train Accuracy ({epochs} epochs)')
                plt.plot(history['val_accuracy'], label=f'Validation Accuracy ({epochs} epochs)')
    
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 6))
    
    for epochs in epochs_list:
        history_file = os.path.join(history_path, f'history_{epochs}_epochs.pkl')
        if os.path.exists(history_file):
            with open(history_file, 'rb') as file:
                history = pickle.load(file)
                plt.plot(history['loss'], label=f'Train Loss ({epochs} epochs)')
                plt.plot(history['val_loss'], label=f'Validation Loss ({epochs} epochs)')
    
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def main():
    data_dir = '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation'
    batch_size = 32
    epochs_list = [100, 200, 500, 1000]
    history_path = 'history_files'
    os.makedirs(history_path, exist_ok=True)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_and_evaluate, epochs, data_dir, batch_size, history_path) for epochs in epochs_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                history, model_file = future.result()
                print(f"Model saved to: {model_file}")
            except Exception as e:
                print(f"Training failed: {e}")

    plot_results(history_path, epochs_list)

if __name__ == "__main__":
    main()

