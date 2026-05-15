import json
import os
import shutil
import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)-8s | %(message)s')
logger = logging.getLogger(__name__)

def get_augmentation_pipeline():
    """Builds the augmentation pipeline as used in the successful Kaggle run."""
    return tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomContrast(0.1),
    ])

def build_model(num_classes):
    """Constructs the optimized MobileNetV3-Small architecture."""
    data_augmentation = get_augmentation_pipeline()
    
    base_model = MobileNetV3Small(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    base_model.trainable = False
    
    inputs = layers.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False) # Force inference mode for BN stability
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs), base_model

def main():
    # Load metadata
    with open('data/class_names.json', 'r') as f:
        names = json.load(f)
    num_classes = len(names)

    # Load datasets
    t_ds = tf.keras.utils.image_dataset_from_directory(
        "data/plant_village/split/train", image_size=(224,224), batch_size=64, seed=42, class_names=names, label_mode='int'
    ).unbatch().batch(64, drop_remainder=True).shuffle(1000).prefetch(tf.data.AUTOTUNE)
    
    v_ds = tf.keras.utils.image_dataset_from_directory(
        "data/plant_village/split/val", image_size=(224,224), batch_size=64, shuffle=False, class_names=names, label_mode='int'
    ).prefetch(tf.data.AUTOTUNE)

    model, base_model = build_model(num_classes)

    # Phase A: Training Head
    logger.info("=== Phase A: Training Head (Frozen Backbone) ===")
    model.compile(optimizer=optimizers.Adam(1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history_a = model.fit(
        t_ds, validation_data=v_ds, epochs=20, 
        callbacks=[
            callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
        ]
    )

    # Phase B: Fine-tuning
    logger.info("=== Phase B: Fine-Tuning (Last 30 Layers) ===")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False
            
    model.compile(optimizer=optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    history_b = model.fit(
        t_ds, validation_data=v_ds, epochs=10,
        callbacks=[
            callbacks.ModelCheckpoint('models/best_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1),
            callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, verbose=1)
        ]
    )

    # Save outputs
    os.makedirs('outputs', exist_ok=True)
    def ser(h): return {k: [float(v) for v in vals] for k, vals in h.items()}
    with open('outputs/training_history.json', 'w') as f:
        json.dump({"phase_a": ser(history_a.history), "phase_b": ser(history_b.history)}, f, indent=2)

    if os.path.exists('models/best_model.keras'):
        shutil.copy('models/best_model.keras', 'models/best_model.h5')
    logger.info("Training complete. Model saved to models/best_model.h5")

if __name__ == "__main__":
    main()
