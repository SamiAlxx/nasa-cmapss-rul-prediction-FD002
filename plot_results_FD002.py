import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from src.data_loader import load_data
from src.features import build_features, assign_conditions, normalize_by_condition, train_val_split
from src.sequence import build_sequences, CMAPSSDataset
from src.models import LSTMModel, CNNModel
from src.evaluate import rmse

torch.manual_seed(0)
np.random.seed(0)

with open('./configs/config_FD002.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg['data']
lstm_cfg = cfg['lstm']
cnn_cfg  = cfg['cnn']

SEQUENCE_LENGTH = data_cfg['sequence_length']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Data pipeline ──────────────────────────────────────────────────────────────
train, test, rul = load_data(data_cfg['data_dir'], subset='FD002')
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

train, val = train_val_split(train)

train, kmeans = assign_conditions(train, n_clusters=data_cfg['n_conditions'])
val,   _      = assign_conditions(val,   n_clusters=data_cfg['n_conditions'], kmeans=kmeans)
test,  _      = assign_conditions(test,  n_clusters=data_cfg['n_conditions'], kmeans=kmeans)

drop_cols    = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL', 'condition']
feature_cols = [c for c in train.columns if c not in drop_cols]

train, val, test, _ = normalize_by_condition(train, val, test, feature_cols)

y_test = rul['RUL'].values

X_test_last = []
for _, group in test.groupby('engine_id'):
    group = group.sort_values('cycle')
    data  = group[feature_cols].values
    if len(data) >= SEQUENCE_LENGTH:
        X_test_last.append(data[-SEQUENCE_LENGTH:])
    else:
        pad = np.zeros((SEQUENCE_LENGTH - len(data), len(feature_cols)))
        X_test_last.append(np.vstack([pad, data]))
X_test_last = np.array(X_test_last, dtype=np.float32)

# ── Load models ────────────────────────────────────────────────────────────────
input_size = X_test_last.shape[2]

lstm_model = LSTMModel(
    input_size=input_size,
    hidden_size=lstm_cfg['hidden_size'],
    num_layers=lstm_cfg['num_layers'],
    dropout=lstm_cfg['dropout']
).to(device)
lstm_model.load_state_dict(torch.load('models/lstm_fd002_best.pth', map_location=device))
lstm_model.eval()

cnn_model = CNNModel(
    input_size=input_size,
    num_filters=cnn_cfg['num_filters'],
    kernel_size=cnn_cfg['kernel_size'],
    dropout=cnn_cfg['dropout']
).to(device)
cnn_model.load_state_dict(torch.load('models/cnn_fd002_best.pth', map_location=device))
cnn_model.eval()

with torch.no_grad():
    X_tensor    = torch.tensor(X_test_last).to(device)
    lstm_preds  = np.clip(lstm_model(X_tensor).cpu().numpy(), 0, 125)
    cnn_preds   = np.clip(cnn_model(X_tensor).cpu().numpy(),  0, 125)

lstm_rmse = rmse(y_test, lstm_preds)
cnn_rmse  = rmse(y_test, cnn_preds)

# ── Plot: Predicted vs True RUL ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, preds, name, score in zip(
    axes,
    [lstm_preds, cnn_preds],
    ['LSTM', '1D CNN'],
    [lstm_rmse, cnn_rmse]
):
    ax.scatter(y_test, preds, alpha=0.4, s=15, color='steelblue')
    lim = max(y_test.max(), preds.max()) + 5
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1, label='Perfect prediction')
    ax.set_xlabel('True RUL')
    ax.set_ylabel('Predicted RUL')
    ax.set_title(f'{name}  —  RMSE: {score:.2f}')
    ax.legend()
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)

plt.suptitle('Predicted vs True RUL — FD002', fontsize=13)
plt.tight_layout()
plt.savefig('plots/rul_prediction_fd002.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved plots/rul_prediction_fd002.png")
