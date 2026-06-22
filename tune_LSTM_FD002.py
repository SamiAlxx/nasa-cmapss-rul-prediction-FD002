import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import optuna

from src.data_loader import load_data
from src.features import build_features, assign_conditions, normalize_by_condition, train_val_split
from src.sequence import build_sequences, CMAPSSDataset
from src.models import LSTMModel
from src.evaluate import rmse

with open('./configs/config_FD002.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train, test, rul = load_data(cfg['data']['data_dir'], subset='FD002')
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

train, val = train_val_split(train)

train, kmeans = assign_conditions(train, n_clusters=cfg['data']['n_conditions'])
val,   _      = assign_conditions(val,   n_clusters=cfg['data']['n_conditions'], kmeans=kmeans)
test,  _      = assign_conditions(test,  n_clusters=cfg['data']['n_conditions'], kmeans=kmeans)

drop_cols    = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL', 'condition']
feature_cols = [c for c in train.columns if c not in drop_cols]

train, val, test, _ = normalize_by_condition(train, val, test, feature_cols)

y_test = rul['RUL'].values

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment('cmapss_fd002_lstm_tuning')

def objective(trial):
    lr          = trial.suggest_float('lr', 5e-4, 5e-3, log=True)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    num_layers  = trial.suggest_int('num_layers', 1, 3)
    dropout     = trial.suggest_float('dropout', 0.1, 0.4)
    seq_len     = trial.suggest_categorical('seq_len', [20, 30, 50])
    batch_size  = trial.suggest_categorical('batch_size', [32, 64, 128])
    patience    = 20

    X_train, y_train = build_sequences(train, feature_cols, seq_len)
    X_val,   y_val   = build_sequences(val,   feature_cols, seq_len)

    train_loader = DataLoader(CMAPSSDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(CMAPSSDataset(X_val,   y_val),   batch_size=batch_size, shuffle=False)

    model = LSTMModel(
        input_size=X_train.shape[2],
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss    = float('inf')
    best_model_state = None
    patience_counter = 0

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            'lr': lr, 'hidden_size': hidden_size, 'num_layers': num_layers,
            'dropout': dropout, 'seq_len': seq_len, 'batch_size': batch_size
        })

        for epoch in range(200):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                loss = criterion(model(X_batch), y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    val_loss += criterion(model(X_batch), y_batch).item()
            val_loss /= len(val_loader)

            if val_loss < best_val_loss:
                best_val_loss    = val_loss
                best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        model.load_state_dict(best_model_state)
        model.eval()

        X_test_last = []
        for _, group in test.groupby('engine_id'):
            group = group.sort_values('cycle')
            data  = group[feature_cols].values
            if len(data) >= seq_len:
                X_test_last.append(data[-seq_len:])
            else:
                pad = np.zeros((seq_len - len(data), len(feature_cols)))
                X_test_last.append(np.vstack([pad, data]))
        X_test_last = np.array(X_test_last, dtype=np.float32)

        with torch.no_grad():
            preds     = np.clip(model(torch.tensor(X_test_last).to(device)).cpu().numpy(), 0, 125)
        test_rmse = rmse(y_test, preds)

        mlflow.log_metric('val_loss', best_val_loss)
        mlflow.log_metric('rmse',     test_rmse)
        print(f"Trial {trial.number} — val_loss: {best_val_loss:.2f} | RMSE: {test_rmse:.2f} | lr={lr:.4f}, hidden={hidden_size}, layers={num_layers}, dropout={dropout:.2f}, seq={seq_len}")

    return best_val_loss

with mlflow.start_run(run_name='optuna_study'):
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=20)

print(f"\nBest val_loss: {study.best_value:.2f}")
print(f"Best params:   {study.best_params}")
