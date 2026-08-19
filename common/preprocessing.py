"""
Shared preprocessing pipeline for all 4 models.

Downloads the IMDB dataset, builds train/val/test tf.data.Dataset objects,
and provides two vectorization layers:
  - int_vectorizer   -> int-encoded, padded sequences (for CNN, LSTM, Attention)
  - tfidf_vectorizer -> TF-IDF vectors (for the MLP baseline)

Run this file directly once to sanity-check the pipeline:
    python -m common.preprocessing
"""

import os
import re
import shutil
import string

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
SEED = 42
MAX_TOKENS = 10000      # vocabulary size
SEQUENCE_LENGTH = 250   # pad/truncate all reviews to this length

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def download_dataset():
    """Downloads and extracts the Stanford aclImdb dataset (first run only;
    cached afterwards). Returns the path to the extracted aclImdb folder."""
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    dataset = tf.keras.utils.get_file(
        "aclImdb_v1.tar.gz",
        url,
        untar=True,
        cache_dir=os.path.abspath(DATA_DIR),
        cache_subdir="",
    )
    base_dir = os.path.dirname(dataset)

    # Keras' extraction folder naming differs across versions/platforms —
    # it may extract directly to <base_dir>/aclImdb, or nest it one level
    # deeper as <base_dir>/aclImdb_v1_extracted/aclImdb. Handle both.
    candidate_paths = [
        os.path.join(base_dir, "aclImdb"),
        os.path.join(base_dir, "aclImdb_v1_extracted", "aclImdb"),
    ]
    dataset_dir = next((p for p in candidate_paths if os.path.isdir(p)), None)
    if dataset_dir is None:
        raise FileNotFoundError(
            f"Could not find extracted aclImdb folder. Checked: {candidate_paths}. "
            f"Check what's actually inside {base_dir}."
        )

    # The archive ships with an unused "unsup" (unlabeled) folder inside
    # train/ — remove it or it gets picked up as a spurious 3rd class.
    unsup_dir = os.path.join(dataset_dir, "train", "unsup")
    if os.path.exists(unsup_dir):
        shutil.rmtree(unsup_dir)

    return dataset_dir


def custom_standardization(input_data):
    """Lowercases text, strips <br /> tags, and removes punctuation."""
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(
        stripped_html, "[%s]" % re.escape(string.punctuation), ""
    )


def get_datasets():
    """Returns (raw_train_ds, raw_val_ds, raw_test_ds) as tf.data.Dataset
    objects yielding (text, label) batches. Train folder is split 80/20 for
    train/val; the official test folder is used as the held-out test set.
    Every model MUST call this same function so all 4 models see the exact
    same split."""
    dataset_dir = download_dataset()
    train_dir = os.path.join(dataset_dir, "train")
    test_dir = os.path.join(dataset_dir, "test")

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, batch_size=BATCH_SIZE, validation_split=0.2,
        subset="training", seed=SEED,
    )
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, batch_size=BATCH_SIZE, validation_split=0.2,
        subset="validation", seed=SEED,
    )
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dir, batch_size=BATCH_SIZE,
    )
    return raw_train_ds, raw_val_ds, raw_test_ds


def build_vectorizers(raw_train_ds):
    """Adapts and returns (int_vectorizer, tfidf_vectorizer) on the
    TRAINING text only (never val/test — that would leak information)."""
    train_text = raw_train_ds.map(lambda x, y: x)

    int_vectorizer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_TOKENS,
        output_mode="int",
        output_sequence_length=SEQUENCE_LENGTH,
    )
    int_vectorizer.adapt(train_text)

    tfidf_vectorizer = tf.keras.layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=MAX_TOKENS,
        output_mode="tf_idf",
    )
    tfidf_vectorizer.adapt(train_text)

    return int_vectorizer, tfidf_vectorizer


def vectorize_datasets(raw_train_ds, raw_val_ds, raw_test_ds, vectorizer):
    """Applies a vectorizer to text in each split. Returns cached,
    prefetching-ready tf.data.Dataset objects of (features, label)."""
    def apply(ds):
        ds = ds.map(lambda x, y: (vectorizer(x), y), num_parallel_calls=AUTOTUNE)
        return ds.cache().prefetch(buffer_size=AUTOTUNE)

    return apply(raw_train_ds), apply(raw_val_ds), apply(raw_test_ds)


if __name__ == "__main__":
    raw_train_ds, raw_val_ds, raw_test_ds = get_datasets()
    int_vec, tfidf_vec = build_vectorizers(raw_train_ds)
    print("Int vocab size:", len(int_vec.get_vocabulary()))
    print("TF-IDF vocab size:", len(tfidf_vec.get_vocabulary()))
    for text_batch, label_batch in raw_train_ds.take(1):
        print("Sample review:", text_batch.numpy()[0][:200])
        print("Sample label:", label_batch.numpy()[0])
