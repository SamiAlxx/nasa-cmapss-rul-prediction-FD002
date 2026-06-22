import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

np.random.seed(0)

from src.data_loader import load_data
from src.features import build_features, assign_conditions, normalize_by_condition, train_val_split
from src.evaluate import rmse, nasa_score

with open('./configs/config_FD002.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

train, test, rul = load_data(cfg['data']['data_dir'], subset='FD002')
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

train, val = train_val_split(train)

train, kmeans = assign_conditions(train, n_clusters=cfg['data']['n_conditions'])
val,   _      = assign_conditions(val,   n_clusters=cfg['data']['n_conditions'], kmeans=kmeans)
test,  _      = assign_conditions(test,  n_clusters=cfg['data']['n_conditions'], kmeans=kmeans)

drop_cols    = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL', 'condition']
feature_cols = [c for c in train.columns if c not in drop_cols]

train, val, test, scalers = normalize_by_condition(train, val, test, feature_cols)

X_train = train[feature_cols]
y_train = train['RUL']
X_test  = test.groupby('engine_id').last().reset_index()[feature_cols]
y_test  = rul['RUL'].values

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_baseline'])

with mlflow.start_run(run_name="xgboost_fd002_baseline"):
    params = cfg['xgboost']
    model  = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)

    preds      = np.clip(model.predict(X_test), 0, 125)
    score_rmse = rmse(y_test, preds)
    score_nasa = nasa_score(y_test, preds)

    mlflow.log_params(params)
    mlflow.log_metric("rmse", score_rmse)
    mlflow.log_metric("nasa_score", score_nasa)

    print(f"RMSE:       {score_rmse:.2f}")
    print(f"NASA Score: {score_nasa:.2f}")
