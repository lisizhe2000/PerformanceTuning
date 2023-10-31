import xgboost as xgb
import numpy as np

# 准备数据
X = np.array([[True, False],
              [False, True],
              [True, True],
              [False, False]]).astype(int)
X = np.random.randint(0, 2, (10, 10))
X = np.random.choice(a=[False, True], size=(10, 10), p=[0.5, 0.5])
y = np.array([1, 0, 1, 0])  # 目标变量，可以是0或1
y = np.random.randint(0, 2, size=10)
print(y)

# # 创建DMatrix
# dmatrix = xgb.DMatrix(X, label=y)

# 定义XGBoost模型
# model = xgb.XGBClassifier(
#     learning_rate=0.1,
#     n_estimators=100
# )

# # 拟合模型
# model.fit(X, y)

# # 进行预测
# predictions = model.predict(np.array(X[0]))

# # 输出预测结果
# print(predictions)

param = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'reg:squarederror'
        }
dtrain = xgb.DMatrix(X, label=y)
model = xgb.train(param, dtrain, 10)
print(model.predict(xgb.DMatrix([X[0]])))
