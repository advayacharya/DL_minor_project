# ICT 4442 — Deep Learning Mini Project

Comparative analysis of MLP, CNN, LSTM, and an Attention-based model on
binary sentiment classification of the IMDB Movie Review dataset.

## Team

- Advay Acharya — MLP, CNN
- Atharv Pawar — LSTM, Attention-based model

## Setup

```bash
pip install -r requirements.txt
```

First run of any model script downloads the IMDB dataset automatically into `data/` (not committed to git — see `.gitignore`).

## Structure

```
common/
  preprocessing.py   # shared dataset loading + vectorization (used by all 4 models)
  utils.py            # shared evaluation/result-saving helper
models/
  mlp/mlp.py           # MLP baseline (TF-IDF input)
  cnn/cnn.py            # CNN (embedding + Conv1D)
  lstm/lstm.py           # LSTM (embedding + LSTM)
  attention/attention.py # Small Transformer encoder block
evaluate/
  compare.py          # builds the final comparison table from all 4 results
reports/
  results/            # each model's metrics, saved as JSON
  (Synopsis/Interim/Final .docx files go here)
```

## Running a model

From the repo root (so the `common` package is importable):

```bash
python -m models.mlp.mlp
python -m models.cnn.cnn
python -m models.lstm.lstm
python -m models.attention.attention
```

Each run trains the model, evaluates it on the held-out test set, and saves
its metrics to `reports/results/<model_name>.json`.

## Comparing all models

Once all 4 have been trained at least once:

```bash
python -m evaluate.compare
```

This prints a single table with accuracy, precision, recall, F1, parameter
count, and training time for all 4 models.
