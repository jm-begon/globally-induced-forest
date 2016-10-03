# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# License: BSD 3 clause
import numpy as np
from time import time

from sklearn.metrics import zero_one_loss as score


from gif.dataset import partition_data, load_hastie as load_data
from gif import GIFClassifier as Estimator


if __name__ == '__main__':
    X_ls, y_ls, X_ts, y_ts = partition_data(load_data(random_state=0))

    est = Estimator(n_estimators=100,
                    budget=5000,
                    learning_rate=.1,
                    random_state=0)

    start = time()
    est.fit(X_ls, y_ls)
    end = time()
    print "Fitting time [s]:", end - start

    start = time()
    predictions = est.predict(X_ts)
    end = time()
    print "Prediction time [s]:", end - start

    ts_score = score(y_ts, predictions)
    ls_score = score(y_ls, est.predict(X_ls))

    print "TS score:", ts_score
    print "LS score", ls_score




