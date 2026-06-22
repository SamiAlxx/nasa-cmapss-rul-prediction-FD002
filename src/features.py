import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml
from sklearn.cluster import KMeans
import numpy as np

with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

DROP_SENSORS = cfg['data']['drop_sensors']
RUL_CAP      = cfg['data']['rul_cap']
WINDOW       = cfg['data']['window']

def train_val_split(df: pd.DataFrame, val_frac: float = 0.2, random_state: int = 42):
    engine_ids = df['engine_id'].unique()
    rng        = np.random.default_rng(random_state)
    val_ids    = rng.choice(engine_ids, size=int(len(engine_ids) * val_frac), replace=False)
    val_mask   = df['engine_id'].isin(val_ids)
    return df[~val_mask].reset_index(drop=True), df[val_mask].reset_index(drop=True)

def add_rul_target(df: pd.DataFrame) -> pd.DataFrame:
    max_cycles = df.groupby('engine_id')['cycle'].max().reset_index()
    max_cycles.columns = ['engine_id', 'max_cycle']
    df = df.merge(max_cycles, on='engine_id')
    df['RUL'] = (df['max_cycle'] - df['cycle']).clip(upper=RUL_CAP)
    df.drop(columns='max_cycle', inplace=True)
    return df

def drop_flat_sensors(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=DROP_SENSORS, errors='ignore')

def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    sensor_cols = [c for c in df.columns if c.startswith('s')]
    df = df.sort_values(['engine_id', 'cycle'])
    for sensor in sensor_cols:
        df[f'{sensor}_mean_{WINDOW}'] = (
            df.groupby('engine_id')[sensor]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
        )
        df[f'{sensor}_std_{WINDOW}'] = (
            df.groupby('engine_id')[sensor]
            .transform(lambda x: x.rolling(WINDOW, min_periods=1).std().fillna(0))
        )
    return df

def build_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    df = drop_flat_sensors(df)
    if is_train:
        df = add_rul_target(df)
    df = add_rolling_features(df)
    return df


def assign_conditions(df, n_clusters=6, kmeans=None):
    op_cols = ['op_1' , 'op_2' , 'op_3']
    if kmeans is None:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        df['condition'] = kmeans.fit_predict(df[op_cols])
    else:
        df['condition'] = kmeans.predict(df[op_cols])
    return df, kmeans 

def normalize_by_condition(train_df, val_df, test_df, feature_cols):
    scalers = {}
    train_df[feature_cols] = train_df[feature_cols].astype(float)
    val_df[feature_cols]   = val_df[feature_cols].astype(float)
    test_df[feature_cols]  = test_df[feature_cols].astype(float)
    for c in sorted(train_df['condition'].unique()):
        scaler     = MinMaxScaler()
        train_mask = train_df['condition'] == c
        val_mask   = val_df['condition'] == c
        test_mask  = test_df['condition'] == c
        train_df.loc[train_mask, feature_cols] = scaler.fit_transform(train_df.loc[train_mask, feature_cols])
        val_df.loc[val_mask, feature_cols]     = scaler.transform(val_df.loc[val_mask, feature_cols])
        test_df.loc[test_mask, feature_cols]   = scaler.transform(test_df.loc[test_mask, feature_cols])
        scalers[c] = scaler
    return train_df, val_df, test_df, scalers