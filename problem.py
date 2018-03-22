import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import ShuffleSplit
import numpy as np

########### We compute our loss functions ################

class BaseScoreType(object):
    def check_y_pred_dimensions(self, y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Wrong y_pred dimensions: y_pred should have {} instances, '
                'instead it has {} instances'.format(len(y_true), len(y_pred)))

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    def score_function(self, ground_truths, predictions, valid_indexes=None):
        if valid_indexes is None:
            valid_indexes = slice(None, None, None)
        y_true = ground_truths.y_pred[valid_indexes]
        y_pred = predictions.y_pred[valid_indexes]
        self.check_y_pred_dimensions(y_true, y_pred)
        return self.__call__(y_true, y_pred)


import numpy as np

class lossBusiness (BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')
    def __init__(self, name='lossBusiness', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        l = 0
        for i in range(len(y_true)):
            if y_pred[i] > 1.1 * y_true[i]:
                l += 0.1 *  y_true[i]
            else:
                l+= 0.1 * abs(y_pred[i] - y_true[i])
        return l

class RMSLE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='rmsle', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        return np.sqrt(np.mean(np.square( np.log(y_pred+1) - np.log(y_true+1) )))


#########################################################


problem_title = 'AirBnB pricing regression'
_target_column_name = 'price'
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_regression()
# An object implementing the workflow
workflow = rw.workflows.FeatureExtractorRegressor()


score_types = [
    RMSLE(),
    lossBusiness(),
    rw.score_types.RMSE(),
    rw.score_types.RelativeRMSE(name='rel_rmse')
]


def get_cv(X, y):
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=57)
    return cv.split(X)


def _read_data(path, f_name):
    data = pd.read_csv(os.path.join(path, 'data', f_name))
    y_array = data[_target_column_name].values
   # y_array = np.log(data[_target_column_name].values)
    X_df = data.drop(_target_column_name, axis=1)
    test = os.getenv('RAMP_TEST_MODE', 0)
    if test:
        return X_df[:100], y_array[:100]
    else:
        return X_df, y_array


def get_train_data(path='.'):
    f_name = 'train.csv'
    return _read_data(path, f_name)


def get_test_data(path='.'):
    f_name = 'test.csv'
    return _read_data(path, f_name)
