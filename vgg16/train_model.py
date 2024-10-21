import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from preprocess import split_data

def build_model(num_classes, input_shape=(256, 256, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

def save_history_plot(history, epochs):
    plt.figure()
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy for {} Epochs'.format(epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot_{}_epochs.png'.format(epochs))
    plt.close()
    
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss for {} Epochs'.format(epochs))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot_{}_epochs.png'.format(epochs))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs to train the model")
    args = parser.parse_args()
    
    epochs = args.epochs
    batch_size = 32
    
    train_generator, val_generator = split_data(
        '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation',
        batch_size=batch_size,
        img_size=(256, 256)
    )
    
    num_classes = len(train_generator.class_indices)
    
    model = build_model(num_classes)
    
    checkpoint = ModelCheckpoint('vgg16_model_{}_epochs.h5'.format(epochs), monitor='val_accuracy', save_best_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, mode='max')
    
    history = model.fit(
        train_generator,
        epochs=epochs,
        validation_data=val_generator,
        callbacks=[checkpoint, early_stopping]
    )
    
    model.save('final_vgg16_model_{}_epochs.h5'.format(epochs))
    save_history_plot(history, epochs)

    # Save history
    history_df = pd.DataFrame(history.history)
    history_df.to_csv('history_{}_epochs.csv'.format(epochs), index=False)

