# Authors: Jean-Michel Begon <jm.begon@gmail.com>
# License: BSD 3 clause
from __future__ import print_function, unicode_literals

from time import time

from sklearn.metrics import zero_one_loss as score


from gif.dataset import partition_data, load_waveform as load_data
from gif import GIFClassifier as Estimator


if __name__ == '__main__':
    random_state = 0
    init_pool_size = 50
    budget = 5000
    learning_rate = .1

    X_ls, y_ls, X_ts, y_ts = partition_data(load_data(random_state=random_state))

    for dynamic_pool in [False, True]:

        print("======= Dynamic Pool: ", dynamic_pool, "========")

        est = Estimator(init_pool_size=init_pool_size,
                        dynamic_pool=dynamic_pool,
                        budget=budget,
                        learning_rate=learning_rate,
                        random_state=random_state)

        print(repr(est))


        start = time()
        est.fit(X_ls, y_ls)
        end = time()
        print("Fitting time [s]:", end - start)

        start = time()
        predictions = est.predict(X_ts)
        end = time()
        print("Prediction time [s]:", end - start)

        ts_score = score(y_ts, predictions)
        ls_score = score(y_ls, est.predict(X_ls))

        print("TS score:", ts_score)
        print("LS score", ls_score)
        print("Actual budget", est.actual_budget)
        print("")
