import pandas as pd
import numpy as np
import yaml
from xgboost import XGBRegressor
import mlflow
import mlflow.sklearn

from src.data_loader import load_data
from src.features import build_features
from src.evaluate import rmse, nasa_score

with open('./configs/config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

train, test, rul = load_data(cfg['data']['data_dir'])
train = build_features(train, is_train=True)
test  = build_features(test,  is_train=False)

drop_cols    = ['engine_id', 'cycle', 'op_1', 'op_2', 'op_3', 'RUL']
feature_cols = [c for c in train.columns if c not in drop_cols]

X_train = train[feature_cols]
y_train = train['RUL']
X_test  = test.groupby('engine_id').last().reset_index()[feature_cols]
y_test  = rul['RUL'].values

mlflow.set_tracking_uri(cfg['mlflow']['tracking_uri'])
mlflow.set_experiment(cfg['mlflow']['experiment_baseline'])

with mlflow.start_run(run_name="xgboost_baseline"):
    params = cfg['xgboost']
    model  = XGBRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        learning_rate=params['learning_rate'],
        random_state=params['random_state']
    )
    model.fit(X_train, y_train)

    preds       = np.clip(model.predict(X_test), 0, 125)
    score_rmse  = rmse(y_test, preds)
    score_nasa  = nasa_score(y_test, preds)

    mlflow.log_params(params)
    mlflow.log_metric("rmse", score_rmse)
    mlflow.log_metric("nasa_score", score_nasa)

    print(f"RMSE:       {score_rmse:.2f}")
    print(f"NASA Score: {score_nasa:.2f}")