# Turbofan Engine RUL Prediction — NASA C-MAPSS FD002

Extension of the FD001 RUL prediction project to the FD002 subset, which introduces 6 distinct operating conditions. This required a fundamentally different preprocessing pipeline compared to FD001.

---

## What Changed from FD001

FD001 has a single operating condition — every engine runs under the same settings, so a single global MinMaxScaler normalizes all sensor readings consistently.

FD002 has 6 operating conditions. Engines shift between discrete operating regimes (different throttle, altitude, and Mach settings) throughout their lifecycle. Applying a global scaler mixes readings from different conditions together, making it impossible for the model to distinguish healthy from degraded behavior. A reading of s2=550 means something very different in condition 0 vs condition 3.

The solution is condition-wise normalization: identify which operating regime each cycle belongs to, then fit a separate scaler per condition on training data only.

Additional changes from FD001:
- **s6 retained**: In FD001, s6 was flat and dropped. Per-condition correlation analysis on FD002 shows s6 correlates consistently with RUL across all 6 conditions (r = -0.10 to -0.69), revealing a degradation signal that was masked by condition-mixing in FD001.
- **Train/validation split**: 20% of training engines held out by engine ID for early stopping. Splitting by engine (not by row) prevents data leakage.
- **Early stopping**: Both LSTM and CNN use validation loss with patience=30 instead of fixed epochs.
- **Hyperparameter tuning**: Optuna used to search over learning rate, hidden size, layers, dropout, sequence length, and batch size using validation loss as the objective.

---

## Dataset

NASA C-MAPSS FD002 subset:
- 260 training engines run to failure
- 259 test engines cut off at an unknown point before failure
- 26 columns per row: engine ID, cycle, 3 operational settings, 21 sensor readings
- **6 operating conditions**, single fault mode

RUL targets are capped at 125 cycles, following the same piecewise linear convention as FD001.

---

## Methodology

### Pipeline

1. **Load data** — raw sensor readings with operational settings
2. **Drop flat sensors** — s1, s5, s10, s16, s18, s19 dropped (near-zero variance across all conditions). s6 retained — see EDA notebook.
3. **Engineer RUL target** — cycles-to-failure, capped at 125
4. **Rolling features** — 30-cycle rolling mean and std added per sensor (28 additional features per engine)
5. **Train/val split** — 80% of engines for training, 20% held out by engine ID
6. **Assign operating conditions** — k-means clustering (k=6) on op_1/op_2/op_3 fitted on train only, applied to val and test
7. **Condition-wise normalization** — separate MinMaxScaler per condition, fitted on train only, applied to val and test

### Models

**XGBoost** trains on the last observed cycle per test engine (tabular, no sequences). Uses all 42 features (14 sensors + 28 rolling features).

**LSTM** — 2 layers, hidden size 64, dropout 0.2, sequence length 30, batch size 64, lr 0.001. Early stopping with patience 30.

**1D CNN** — 64 filters, kernel size 3, dropout 0.2, sequence length 30, batch size 64, lr 0.001. Early stopping with patience 30.

All runs use `random_seed=0` for reproducibility.

---

## Results

| Model | RMSE | NASA Score |
|-------|------|------------|
| XGBoost | 29.69 | 12,322 |
| 1D CNN | 30.02 | — |
| **LSTM** | **26.00** | **~7,300** |

Lower is better for both metrics. LSTM is the best performer on FD002, unlike FD001 where CNN won.

FD002 results are significantly worse than FD001 across all models (FD001 best: CNN RMSE 15.03). This is expected — 6 operating conditions make the prediction problem substantially harder, and the validation set (52 engines across 6 conditions) is small enough to introduce instability in early stopping.

---

## Key Findings

**Condition-wise normalization is essential.** Without it, a single scaler conflates sensor readings from different operating regimes, destroying the model's ability to detect degradation.

**K-means on operational settings cleanly separates the 6 conditions.** The 3 operational settings (op_1, op_2, op_3) form 6 well-separated clusters with minimal overlap, making k-means a reliable assignment method.

**s6 is condition-dependent.** It appeared useless in FD001 (flat signal, dropped) but shows consistent RUL correlation in FD002 once you look within each condition separately. This illustrates a broader point: sensor usefulness can only be assessed after accounting for operating regime.

**LSTM outperforms CNN on FD002.** The reversed ranking compared to FD001 likely reflects the longer-range temporal dependencies in multi-condition data — the LSTM's gating mechanism handles the condition-switching patterns better than the CNN's local convolutions.

**Validation set noise limits tuning.** With only ~52 val engines split across 6 conditions (~8-9 per condition), the validation loss is noisy. Optuna could not find a reliably better configuration than the baseline — the best trial by val loss gave RMSE 32, while the best by test RMSE was 24.7 but not reproducible. A larger val set or k-fold cross-validation would be needed for reliable hyperparameter optimization.

---

## How to Run

```bash
python train_baseline_FD002.py   # XGBoost
python train_LSTM_FD002.py       # LSTM
python train_cnn_FD002.py        # 1D CNN
python tune_LSTM_FD002.py        # Optuna tuning (20 trials)
```

Hyperparameters can be adjusted in `configs/config_FD002.yaml`.

```bash
mlflow ui
```

Then open http://127.0.0.1:5000 to view all experiments.

---

## Limitations

- Small validation set (52 engines) makes early stopping and hyperparameter tuning noisy
- No cross-validation — results are on a single train/test split
- FD003 and FD004 not yet evaluated
