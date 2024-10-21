import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
dataset_path = '/home/pkalluri/Leaf_diseases/Datasets/plant_village/Plant_leave_diseases_dataset_without_augmentation'

# Prepare the data generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # Split ratio for training and validation data
)

train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Loop through different epoch values
epoch_values = [100, 200, 500, 1000]  # Add or change epoch values as needed

for epochs in epoch_values:
    # Define ResNet-50 model
    base_model = tf.keras.applications.ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    x = base_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    predictions = tf.keras.layers.Dense(train_generator.num_classes, activation='softmax')(x)

    model = tf.keras.Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )

    # Save the model
    model_save_path = f'resnet50_model_{epochs}_epochs.h5'
    model.save(model_save_path)

    print(f"Model trained for {epochs} epochs saved to {model_save_path}")

