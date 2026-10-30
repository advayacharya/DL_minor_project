"""
LSTM — sequential model over word-embedding sequences.
Owner: Atharv Pawar
Run from repo root: python -m models.lstm.lstm
"""

import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import tensorflow as tf

from common.preprocessing import (
    MAX_TOKENS,
    SEQUENCE_LENGTH,
    build_vectorizers,
    get_datasets,
    vectorize_datasets,
)
from common.utils import evaluate_and_save

MODEL_NAME = "lstm"
EPOCHS = 10  # shared across all four models; early stopping ends runs sooner
EMBEDDING_DIM = 64


def build_model(vocab_size, seq_len):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(seq_len,)),
        tf.keras.layers.Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True),
        tf.keras.layers.LSTM(64),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def main():
    raw_train_ds, raw_val_ds, raw_test_ds = get_datasets()
    int_vectorizer, _ = build_vectorizers(raw_train_ds)

    train_ds, val_ds, test_ds = vectorize_datasets(
        raw_train_ds, raw_val_ds, raw_test_ds, int_vectorizer
    )

    model = build_model(MAX_TOKENS, SEQUENCE_LENGTH)
    model.summary()

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )
    start = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS,
              callbacks=[early_stop])
    train_seconds = time.time() - start

    evaluate_and_save(MODEL_NAME, model, test_ds, train_seconds, model.count_params())


if __name__ == "__main__":
    main()
