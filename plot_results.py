import numpy as np
import matplotlib.pyplot as plt
import torch

from src.data_loader import load_data
from src.features import build_features, normalize_features
from src.models import CNNModel, LSTMModel
from src.evaluate import rmse, nasa_score

# --- Load and prepare ---
train, test, rul = load_data('./CMAPSSData')
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

drop_cols = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL']
feature_cols = [c for c in train.columns if c not in drop_cols]
train, test, scaler = normalize_features(train, test, feature_cols)

test['RUL'] = 0
X_test_last = []
for engine_id, group in test.groupby('engine_id'):
    group = group.sort_values('cycle')
    data = group[feature_cols].values
    if len(data) >= 30:
        X_test_last.append(data[-30:])
    else:
        pad = np.zeros((30 - len(data), len(feature_cols)))
        X_test_last.append(np.vstack([pad, data]))

X_test_last = np.array(X_test_last, dtype=np.float32)
y_test = rul['RUL'].values

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Load CNN ---
cnn_model = CNNModel(input_size=X_test_last.shape[2]).to(device)
cnn_model.load_state_dict(torch.load('models/cnn_50epochs.pth'))
cnn_model.eval()

# --- Load LSTM ---
lstm_model = LSTMModel(input_size=X_test_last.shape[2]).to(device)
lstm_model.load_state_dict(torch.load('models/lstm_50epochs.pth'))
lstm_model.eval()

# --- Predict ---
with torch.no_grad():
    X_tensor = torch.tensor(X_test_last).to(device)
    cnn_preds  = np.clip(cnn_model(X_tensor).cpu().numpy(), 0, 125)
    lstm_preds = np.clip(lstm_model(X_tensor).cpu().numpy(), 0, 125)

# --- Scatter plots ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, preds, title in zip(axes,
                             [cnn_preds, lstm_preds],
                             ['CNN (50 epochs)', 'LSTM (50 epochs)']):
    ax.scatter(y_test, preds, alpha=0.6, edgecolors='k', linewidths=0.3)
    ax.plot([0, 125], [0, 125], 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{title} — RMSE: {rmse(y_test, preds):.2f}')
    ax.legend()

plt.tight_layout()
plt.savefig('plots/rul_prediction.png', dpi=150)
plt.show()

print(f"CNN  RMSE: {rmse(y_test, cnn_preds):.2f}")
print(f"LSTM RMSE: {rmse(y_test, lstm_preds):.2f}")