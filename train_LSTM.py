import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from src.data_loader import load_data
from src.features import build_features, normalize_features
from src.sequence import build_sequences, CMAPSSDataset
from src.models import LSTMModel
from src.evaluate import rmse, nasa_score

with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

data_cfg = cfg['data']
lstm_cfg = cfg['lstm']

SEQUENCE_LENGTH = data_cfg['sequence_length']
EPOCHS          = lstm_cfg['epochs']
BATCH_SIZE      = lstm_cfg['batch_size']
LEARNING_RATE   = lstm_cfg['learning_rate']
HIDDEN_SIZE     = lstm_cfg['hidden_size']
NUM_LAYERS      = lstm_cfg['num_layers']
DROPOUT         = lstm_cfg['dropout']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train, test, rul = load_data(data_cfg['data_dir'])
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

drop_cols    = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL']
feature_cols = [c for c in train.columns if c not in drop_cols]

train, test, _ = normalize_features(train, test, feature_cols)

X_train, y_train = build_sequences(train, feature_cols, SEQUENCE_LENGTH)
print(f"X_train shape: {X_train.shape}")

X_test_last = []
for engine_id, group in test.groupby('engine_id'):
    group = group.sort_values('cycle')
    data  = group[feature_cols].values
    if len(data) >= SEQUENCE_LENGTH:
        X_test_last.append(data[-SEQUENCE_LENGTH:])
    else:
        pad = np.zeros((SEQUENCE_LENGTH - len(data), len(feature_cols)))
        X_test_last.append(np.vstack([pad, data]))

X_test_last = np.array(X_test_last, dtype=np.float32)
y_test      = rul['RUL'].values

train_loader = DataLoader(CMAPSSDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_deep'])

with mlflow.start_run(run_name=f"lstm_{EPOCHS}epochs"):
    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    mlflow.log_params({**lstm_cfg, "sequence_length": SEQUENCE_LENGTH})

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        mlflow.log_metric("train_loss", avg_loss, step=epoch)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

    model.eval()
    with torch.no_grad():
        preds = np.clip(model(torch.tensor(X_test_last).to(device)).cpu().numpy(), 0, 125)

    score_rmse = rmse(y_test, preds)
    score_nasa = nasa_score(y_test, preds)

    mlflow.log_metric("rmse", score_rmse)
    mlflow.log_metric("nasa_score", score_nasa)

    print(f"\nLSTM Results:")
    print(f"RMSE:       {score_rmse:.2f}")
    print(f"NASA Score: {score_nasa:.2f}")

    torch.save(model.state_dict(), f'models/lstm_{EPOCHS}epochs.pth')
