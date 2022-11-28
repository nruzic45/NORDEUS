import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def randfor(nordeus_train_prepared, nordeus_labels_tr, nordeus_val_prepared, nordeus_labels_vl):
    randFOR = RandomForestRegressor(max_depth=2, n_estimators=120)
    randFOR.fit(nordeus_train_prepared, nordeus_labels_tr)

    errors = [mean_squared_error(nordeus_labels_vl, y_pred)
              for y_pred in randFOR.staged_predict(nordeus_val_prepared)]

    bst_n_estimators = np.argmin(errors) + 1
    randFOR_best = RandomForestRegressor(max_depth=2, n_estimators=bst_n_estimators)
    randFOR_best.fit(nordeus_train_prepared, nordeus_labels_tr)

    print(errors)
    return randFOR_best


