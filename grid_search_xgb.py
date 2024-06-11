import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

task = sys.argv[1]
num_cores = 64
num_fold = 10
seed = 42

params = {
    'n_estimators': [10, 100, 500, 800],
    'max_depth': [2, 3, 4, 5],
    'max_leaves': [2, 3, 4, 5, 6, 7, 8],
    'gamma': [0.5, 1, 1.5, 2, 5],
    'min_child_weight': [1, 5, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
}

if task == 'cls':
    model = XGBClassifier(silent=True, nthread=1, verbosity=0)
    y_col = 'dx'
else:
    model = XGBRegressor(silent=True, nthread=1, verbosity=0)
    y_col = 'mmse'

label_col = ['dx', 'mmse']
sex_mapping = {'F': 0, 'M': 1}
dx_mapping = {'NC': 0, 'MCI': 1}

disvoice_feature_names = ['static-Articulation', 'static-Phonation',
       'static-RepLearning', 'static-Prosody', 'static-Phonological',
       'dynamic-Articulation', 'dynamic-Phonation', 'dynamic-RepLearning',
       'dynamic-Prosody', 'dynamic-Phonological']

# Read data
data = pd.read_csv('/data/datasets/TAUKADIAL-24/train/groundtruth.csv')
data.sex = data.sex.apply(sex_mapping.get)
data.dx = data.dx.apply(dx_mapping.get)
disvoice = pd.read_parquet('/data/datasets/TAUKADIAL-24/feature/feats_train.parquet')
for i in disvoice_feature_names:
    data[i+'_mean'] = disvoice[i].apply(np.mean).fillna(0)
data = data.drop('tkdname', axis=1)

kv = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=seed)

# Grid search
grid_search = GridSearchCV(
    model, 
    param_grid=params, 
    scoring='f1_macro' if task == 'cls' else 'neg_mean_squared_error', 
    n_jobs=num_cores, 
    cv=kv.split(data.drop(label_col, axis=1), data[y_col]), 
    verbose=1)

print('Start:', datetime.now())
grid_search.fit(data.drop(label_col, axis=1), data[y_col])
print('End:', datetime.now())

print('Best score:', grid_search.best_score_)

import pickle
with open(f'outputs/xgb_{task}_best_params.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)
