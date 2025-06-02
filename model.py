import tensorflow as tf
from tensorflow.keras.applications import Xception, ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

def build_deepfake_detector_resnet50(input_shape=(128, 128, 3), learning_rate=0.0001):

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    x = Dense(256, activation='relu', name='fc1_relu')(x)

    x = Dropout(0.5, name='dropout_1')(x) 

    predictions = Dense(1, activation='sigmoid', name='output_sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

def build_deepfake_detector_xception(input_shape=(128, 128, 3), learning_rate=0.0001):
    base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)

    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output

    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    x = Dense(256, activation='relu', name='fc1_relu')(x)

    x = Dropout(0.5, name='dropout_1')(x) 

    predictions = Dense(1, activation='sigmoid', name='output_sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == '__main__':
    DATA_DIR = "data_pre"
    IMG_HEIGHT = 128
    IMG_WIDTH = 128
    CHANNELS = 3
    BATCH_SIZE = 32
    EPOCHS = 10

    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/train",
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=42
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        f"{DATA_DIR}/val",
        labels="inferred",
        label_mode="binary",
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    choice = input("Choose model (1 for Xception, 2 for ResNet50): ")
    if choice =='1':
        # Preprocess images for Xception
        preprocess = tf.keras.applications.xception.preprocess_input
        train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
        val_ds = val_ds.map(lambda x, y: (preprocess(x), y))
        input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        model = build_deepfake_detector_resnet50(input_shape=input_shape)
        model.summary()
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds
            )
    elif choice == '2':
        # Preprocess images for ResNet50
        preprocess = tf.keras.applications.resnet50.preprocess_input
        train_ds = train_ds.map(lambda x, y: (preprocess(x), y))
        val_ds = val_ds.map(lambda x, y: (preprocess(x), y))

        # Build and train model
        input_shape = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)
        model = build_deepfake_detector_resnet50(input_shape=input_shape)
        model.summary()
        history = model.fit(
            train_ds,
            epochs=EPOCHS,
            validation_data=val_ds
            )   
