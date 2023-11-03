from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import numpy as np


class TestML(object):

    def __init__(self) -> None:
        self.X = np.random.choice(a=[False, True], size=(10, 10), p=[0.5, 0.5])
        self.y = np.random.rand(10)
        # self.y = np.random.randint(0, 2, size=10)

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
        
    def test_random_forest(self):
        model = RandomForestRegressor()
        model.fit(self.X, self.y)
        vals = []
        for t in model.estimators_:
            vals.append(t.predict(self.X))
        mean = sum(vals) / len(vals)
        pred = model.predict(self.X)
        print(f'vals: {vals}\nmean: {mean}\npred: {pred}\nnum_trees: {len(model.estimators_)}\ncoefficient of decision: {model.score(self.X, self.y)}')

if __name__ == '__main__':
    test = TestML()
    # test.test_xgboost()
    # test.test_cart()
    test.test_random_forest()
