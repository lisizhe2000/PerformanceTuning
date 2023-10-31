from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np


class TestML(object):

    def __init__(self) -> None:
        self.X = np.random.choice(a=[False, True], size=(10, 10), p=[0.5, 0.5])
        self.y = np.random.randint(0, 2, size=10)

    def test_xgboost(self):
        param = {
            'max_depth': 2,
            'eta': 1,
            'objective': 'reg:squarederror'
        }
        dtrain = xgb.DMatrix(self.X, label=self.y)
        model = xgb.train(param, dtrain, 10)
        print(model.predict(xgb.DMatrix([self.X[0]])))

    def test_cart(self):
        model = DecisionTreeRegressor()
        model.fit(self.X, self.y)
        prediction = model.predict([self.X[6]])
        print(self.y)
        print(prediction)
        
if __name__ == '__main__':
    test = TestML()
    # test.test_xgboost()
    test.test_cart()
