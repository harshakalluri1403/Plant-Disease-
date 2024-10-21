import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from preprocess import preprocess_data

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

def train_model(epochs, data_dir, batch_size):
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
    history_file = f'history_{epochs}_epochs.pkl'
    with open(history_file, 'wb') as file:
        pickle.dump(history.history, file)
    
    # Save the model
    model_file = f'resnet50_leaf_disease_model_{epochs}_epochs.h5'
    model.save(model_file)

if __name__ == "__main__":
    data_dir = '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation'
    batch_size = 32
    epochs = 100
    train_model(epochs, data_dir, batch_size)

