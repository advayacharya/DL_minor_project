# Training curves — MLP & CNN (Advay)

Answers open questions 1 & 2 in training_curves.md.
Python 3.12.10, TensorFlow 2.21.0, CPU. Dataset: original Stanford tarball.
Shared pipeline unchanged. EarlyStopping(val_loss, patience=2,
restore_best_weights=True), max 10 epochs — same as LSTM/Attention.

## MLP

| epoch | train_acc | train_loss | val_acc | val_loss |
|------:|----------:|-----------:|--------:|---------:|
| 1 | 0.8241 | 0.4066 | **0.8904** | **0.2777** |
| 2 | 0.9257 | 0.1987 | 0.8844 | 0.3116 |
| 3 | 0.9574 | 0.1190 | 0.8790 | 0.3701 |

Best val at epoch 1 (restored). Test accuracy: 0.8743 (was 0.8584).

## CNN

| epoch | train_acc | train_loss | val_acc | val_loss |
|------:|----------:|-----------:|--------:|---------:|
| 1 | 0.7681 | 0.4587 | 0.8692 | 0.3117 |
| 2 | 0.9146 | 0.2211 | **0.8810** | **0.2997** |
| 3 | 0.9717 | 0.0916 | 0.8780 | 0.3641 |
| 4 | 0.9880 | 0.0382 | 0.8718 | 0.4570 |

Best val at epoch 2 (restored). Test accuracy: 0.8704 (was 0.8598).

## Reading

Same pattern as LSTM and Attention: val loss bottoms out at epoch 1–2 and
rises after while train loss keeps falling, so both models overfit early.
The old fixed 10-epoch runs were evaluating overfit checkpoints.
All four models now train under identical conditions.
