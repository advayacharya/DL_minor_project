# Training curves — LSTM & Attention (Atharv)

Run on 2026-09-22. TensorFlow 2.21.0 / Keras 3.15.1, Python 3.13, CPU.
Shared pipeline unchanged (`common/preprocessing.py`, SEED=42,
MAX_TOKENS=10000, SEQUENCE_LENGTH=250, BATCH_SIZE=32).
Both models: `model.fit(epochs=6)`, no callbacks.

## LSTM

| epoch | train_acc | train_loss | val_acc | val_loss |
|------:|----------:|-----------:|--------:|---------:|
| 1 | 0.7755 | 0.4640 | 0.8422 | 0.3803 |
| 2 | 0.8932 | 0.2747 | **0.8660** | **0.3209** |
| 3 | 0.9278 | 0.1962 | 0.8626 | 0.3574 |
| 4 | 0.9435 | 0.1584 | 0.8464 | 0.3968 |
| 5 | 0.9495 | 0.1359 | 0.8438 | 0.5214 |
| 6 | 0.9601 | 0.1123 | 0.8168 | 0.5696 |

Best val at epoch 2. Evaluated checkpoint = epoch 6.
Test accuracy as saved: 0.8022

## Attention

| epoch | train_acc | train_loss | val_acc | val_loss |
|------:|----------:|-----------:|--------:|---------:|
| 1 | 0.7638 | 0.4447 | **0.8792** | **0.2963** |
| 2 | 0.9101 | 0.2292 | 0.8754 | 0.3206 |
| 3 | 0.9402 | 0.1599 | 0.8558 | 0.4506 |
| 4 | 0.9566 | 0.1219 | 0.8630 | 0.5283 |
| 5 | 0.9629 | 0.1028 | 0.8598 | 0.5766 |
| 6 | 0.9679 | 0.0879 | 0.8526 | 0.6282 |

Best val at epoch 1. Evaluated checkpoint = epoch 6.
Test accuracy as saved: 0.8300

## Reading

Validation loss reaches its minimum at epoch 2 (LSTM) and epoch 1
(Attention) and rises monotonically after, while training loss keeps
falling. Both models are overfitting well before the fixed 6-epoch
schedule ends, and the checkpoint that gets evaluated and written to
`reports/results/*.json` is the worst one produced.

Attention's epoch-1 val accuracy (0.8792) is higher than any final
number in the project, MLP included (0.8584).

## Open questions for Advay

1. Do MLP and CNN show the same pattern? Need per-epoch val_accuracy
   and val_loss, not just the final figure.
2. Add `EarlyStopping(monitor="val_loss", restore_best_weights=True)`
   to all four models and re-run? Must be all four to keep the
   "identical training conditions" claim in the synopsis.
3. `models/attention/attention.py` — `PositionalEmbedding.token_emb`
   has no `mask_zero=True`, unlike the LSTM's Embedding. Self-attention
   and GlobalAveragePooling1D therefore both run over padding tokens.
