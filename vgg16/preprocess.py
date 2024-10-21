import os
import shutil
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

def split_data(dataset_dir, validation_split=0.2, batch_size=32, img_size=(256, 256)):
    data_gen = ImageDataGenerator(rescale=1.0/255.0, validation_split=validation_split)

    train_generator = data_gen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    val_generator = data_gen.flow_from_directory(
        dataset_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, val_generator

if __name__ == "__main__":
    dataset_dir = '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation'
    train_generator, val_generator = split_data(dataset_dir)
    print(f"Found {train_generator.samples} training images belonging to {train_generator.num_classes} classes.")
    print(f"Found {val_generator.samples} validation images belonging to {val_generator.num_classes} classes.")

