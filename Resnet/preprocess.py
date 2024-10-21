import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data(data_dir, image_size=(256, 256), batch_size=32, validation_split=0.2):
    # Define ImageDataGenerators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split  # Use this to split data into train/validation
    )
    
    valid_datagen = ImageDataGenerator(rescale=1./255, validation_split=validation_split)

    # Load training and validation data
    train_generator = train_datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'  # Set this to use training data
    )
    
    valid_generator = valid_datagen.flow_from_directory(
        directory=data_dir,
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'  # Set this to use validation data
    )
    
    return train_generator, valid_generator

if __name__ == "__main__":
    data_dir = '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation'
    train_gen, valid_gen = preprocess_data(data_dir)

