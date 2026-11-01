"""
MLP baseline — trained on TF-IDF features.
Owner: Advay Acharya
Run from repo root: python -m models.mlp.mlp
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tensorflow as tf

from common.preprocessing import build_vectorizers, get_datasets, vectorize_datasets
from common.utils import evaluate_and_save

MODEL_NAME = "mlp"
EPOCHS = 10  # shared across all four models; early stopping ends runs sooner


def build_model(input_dim):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def print_history(history):
    h = history.history
    print("| epoch | train_acc | train_loss | val_acc | val_loss |")
    print("|------:|----------:|-----------:|--------:|---------:|")
    for i in range(len(h["loss"])):
        print(f"| {i+1} | {h['accuracy'][i]:.4f} | {h['loss'][i]:.4f} "
              f"| {h['val_accuracy'][i]:.4f} | {h['val_loss'][i]:.4f} |")


def main():
    raw_train_ds, raw_val_ds, raw_test_ds = get_datasets()
    _, tfidf_vectorizer = build_vectorizers(raw_train_ds)

    train_ds, val_ds, test_ds = vectorize_datasets(
        raw_train_ds, raw_val_ds, raw_test_ds, tfidf_vectorizer
    )

    vocab_size = len(tfidf_vectorizer.get_vocabulary())
    model = build_model(vocab_size)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
                        callbacks=[early_stop])
    train_seconds = time.time() - start
    print_history(history)

    evaluate_and_save(MODEL_NAME, model, test_ds, train_seconds, model.count_params())


if __name__ == "__main__":
    main()
