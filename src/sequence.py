import numpy as np
import torch
from torch.utils.data import Dataset

SEQUENCE_LENGTH = 30

def build_sequences(df, feature_cols, sequence_length=SEQUENCE_LENGTH):
    X, y = [], []

    for engine_id, group in df.groupby('engine_id'):
        group = group.sort_values('cycle')
        data = group[feature_cols].values
        rul = group['RUL'].values

        for i in range(len(data) - sequence_length + 1):
            X.append(data[i:i + sequence_length])
            y.append(rul[i + sequence_length - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_val_split(df, val_engines=20):
    all_engines = df['engine_id'].unique()
    val_ids     = all_engines[-val_engines:]
    train_ids   = all_engines[:-val_engines]
    
    train_df = df[df['engine_id'].isin(train_ids)].copy()
    val_df   = df[df['engine_id'].isin(val_ids)].copy()
    
    return train_df, val_df


class CMAPSSDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

