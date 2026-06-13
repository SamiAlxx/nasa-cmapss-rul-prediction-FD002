import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yaml

with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

DROP_SENSORS = cfg['data']['drop_sensors']
RUL_CAP      = cfg['data']['rul_cap']
WINDOW       = cfg['data']['window']

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

def normalize_features(train_df, test_df, feature_cols):
    scaler = MinMaxScaler()
    train_df[feature_cols] = scaler.fit_transform(train_df[feature_cols])
    test_df[feature_cols]  = scaler.transform(test_df[feature_cols])
    return train_df, test_df, scaler