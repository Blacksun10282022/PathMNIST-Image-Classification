
from dataclasses import dataclass
from typing import Dict, Any
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_cnn(input_shape, num_classes, arch: Dict[str, Any]):
    x_in = keras.Input(shape=input_shape)
    x = x_in
    for block in arch.get("conv_blocks", [
        {"filters":32, "kernel_size":3, "pool":2, "dropout":0.0},
        {"filters":64, "kernel_size":3, "pool":2, "dropout":0.0},
    ]):
        x = layers.Conv2D(block["filters"], block["kernel_size"], padding="same", activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(block["pool"])(x)
        if block.get("dropout", 0.0) > 0:
            x = layers.Dropout(block["dropout"])(x)
    x = layers.Flatten()(x)
    if arch.get("dense_units", 128) > 0:
        reg = regularizers.l2(arch.get("l2", 0.0)) if arch.get("l2", 0.0) else None
        x = layers.Dense(arch.get("dense_units", 128), activation="relu", kernel_regularizer=reg)(x)
        if arch.get("dropout", 0.2) > 0:
            x = layers.Dropout(arch.get("dropout", 0.2))(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(x_in, out)

def train_cnn(X_tr, y_tr, X_val, y_val, arch, batch_size=128, epochs=20, patience=5, lr=1e-3, out_dir="outputs"):
    num_classes = int(len(set(y_tr.tolist())))
    model = build_cnn(X_tr.shape[1:], num_classes, arch)

    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=max(1, patience//2), verbose=1),
        keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dir, "best.keras"), monitor="val_accuracy", mode="max", save_best_only=True),
    ]

    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        callbacks=callbacks,
    )
    return model, history

def evaluate(model, X, y):
    import numpy as np
    y_pred = model.predict(X, verbose=0).argmax(axis=1)
    return y_pred
