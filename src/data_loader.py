from sys import _enablelegacywindowsfsencoding

import pandas as pd

COLS = (
    ['engine_id', 'cycle'] +
    [f'op_{i}' for i in range(1, 4)] +
    [f's{i}' for i in range(1, 22)]
)

def load_data(data_dir: str, subset: str = 'FD001'):
    train = pd.read_csv(f'{data_dir}/train_{subset}.txt', sep=r'\s+', header=None, names=COLS, engine='python')
    test = pd.read_csv(f'{data_dir}/test_{subset}.txt', sep=r'\s+', header=None, names=COLS, engine='python')
    rul = pd.read_csv(f'{data_dir}/RUL_{subset}.txt', sep=r'\s+', header=None, names=['RUL'], engine='python')
    return train, test, rul
