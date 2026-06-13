import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out.squeeze(1)
    

class CNNModel(nn.Module):
    def __init__(self, input_size, num_filters=64, kernel_size=3, dropout=0.2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters * 2, kernel_size, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(num_filters * 2, 1)

    def forward(self, x):
        # x shape: (batch, sequence, features) → need (batch, features, sequence) for Conv1d
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.mean(dim=2)  # global average pooling
        x = self.dropout(x)
        x = self.fc(x)
        return x.squeeze(1)
    

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.1):
        self.patience  = patience
        self.min_delta = min_delta
        self.counter   = 0
        self.best_loss = None
        self.stop      = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        else:
            self.best_loss = val_loss
            self.counter   = 0