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


def main():
    raw_train_ds, raw_val_ds, raw_test_ds = get_datasets()
    _, tfidf_vectorizer = build_vectorizers(raw_train_ds)

    train_ds, val_ds, test_ds = vectorize_datasets(
        raw_train_ds, raw_val_ds, raw_test_ds, tfidf_vectorizer
    )

    vocab_size = len(tfidf_vectorizer.get_vocabulary())
    model = build_model(vocab_size)
    model.summary()

    start = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=10)
    train_seconds = time.time() - start

    evaluate_and_save(MODEL_NAME, model, test_ds, train_seconds, model.count_params())


if __name__ == "__main__":
    main()
