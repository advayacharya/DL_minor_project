"""
Attention-based model — a small Transformer encoder block over word +
positional embeddings, followed by pooling and a classifier head.
Owner: Atharv Pawar
Run from repo root: python -m models.attention.attention
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

MODEL_NAME = "attention"
EMBEDDING_DIM = 64
NUM_HEADS = 2
FF_DIM = 64


class PositionalEmbedding(tf.keras.layers.Layer):
    """Combines a token embedding with a learned positional embedding, since
    (unlike an LSTM) attention has no built-in notion of word order."""

    def __init__(self, seq_len, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_emb = tf.keras.layers.Embedding(vocab_size, embed_dim)
        self.pos_emb = tf.keras.layers.Embedding(seq_len, embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        return self.token_emb(x) + self.pos_emb(positions)


class TransformerBlock(tf.keras.layers.Layer):
    """One self-attention + feed-forward block, with residual connections
    and layer normalization (the core repeating unit of a Transformer)."""

    def __init__(self, embed_dim, num_heads, ff_dim, **kwargs):
        super().__init__(**kwargs)
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(0.1)
        self.dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, x, training=False):
        attn_out = self.att(x, x)
        x = self.norm1(x + self.dropout1(attn_out, training=training))
        ffn_out = self.ffn(x)
        return self.norm2(x + self.dropout2(ffn_out, training=training))


def build_model(vocab_size, seq_len):
    inputs = tf.keras.layers.Input(shape=(seq_len,))
    x = PositionalEmbedding(seq_len, vocab_size, EMBEDDING_DIM)(inputs)
    x = TransformerBlock(EMBEDDING_DIM, NUM_HEADS, FF_DIM)(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
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

    start = time.time()
    model.fit(train_ds, validation_data=val_ds, epochs=6)
    train_seconds = time.time() - start

    evaluate_and_save(MODEL_NAME, model, test_ds, train_seconds, model.count_params())


if __name__ == "__main__":
    main()
