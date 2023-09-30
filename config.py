from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

rf_params = {"max_depth": [8, 15, 22],
             "max_features": [5, 7, "sqrt", "auto"],
             "min_samples_split": [20, 29, 39],
             "n_estimators": [100, 200]}

xgboost_params = {"learning_rate": [0.01, 0.05, 0.1],
                  "max_depth": range(1, 10),
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.3, 0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.05, 0.1],
                   "n_estimators": [250, 300, 350],
                   "colsample_bytree": [0.8, 1]}

regressor = [("RF", RandomForestRegressor(), rf_params),
               ('XGBoost', XGBRegressor(eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMRegressor(force_col_wise=True, verbose=-1), lightgbm_params)]

