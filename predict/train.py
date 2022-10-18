import joblib
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from xgboost import plot_importance
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
import pandas as pd
import numpy as np

if __name__ == '__main__':
    class_le = LabelEncoder()
    df = pd.read_csv("../data/vehiclemonitorinfo.csv")
    data = df[["vehicleType","vehicleSpeed","vehicleCoor1","vehicleCoor2","conflictX","conflictY","recordTime"]]
    data['vehicleType'] = class_le.fit_transform(data['vehicleType'])
    X = data[["vehicleType","vehicleSpeed","vehicleCoor1","vehicleCoor2","conflictX","conflictY"]].values
    y = data[["recordTime"]].values
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=0)
    xgb_model = XGBRegressor()
    xgb_model.fit(x_train, y_train)
    y_pred = xgb_model.predict(x_test)
    print('MSE为：', mean_squared_error(y_test, y_pred))
    print('RMSE为：', np.sqrt(mean_squared_error(y_test, y_pred)))
    joblib.dump(xgb_model, "xgb_model.pkl")
    # para = {
    #         'learning_rate':[0.3,0.1,0.05,0.03,0.01],
    #         'n_estimators':np.arange(50,200,20),
    #         'max_depth':np.arange(3,10,1),
    #         'gamma':np.arange(0.1,1,.1),
    #         'min_child_weight':np.arange(0,5,1),
    #         'subsample':np.arange(0.3,1,.1),
    #         'colsample_bytree':np.arange(0.3,1,.1),
    #         'colsample_bylevel':np.arange(0.3,1,.1),
    #         'reg_alpha':np.arange(0,0.5,.1),
    #         'reg_lambda':np.arange(0.5,1,.1),
    # }
    # cv = GridSearchCV(estimator=xgb_model, param_grid=para, scoring='neg_mean_squared_error', cv=5, verbose=1, n_jobs=-1)
    # cv.fit(x_train, y_train)
    # print("Best params:", cv.best_params_)
    # print('Best score:', cv.best_score_)
