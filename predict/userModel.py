import sys

import joblib

if __name__ == '__main__':
    model = joblib.load("xgb_model.pkl")  # 加载模型
    value = []
    # 接收参数
    for i in range(len(sys.argv)):
        value.append(float(sys.argv[i]))
    # 预测
    predict = model.predict(value)
