import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error


def gbr(nordeus_train_prepared, nordeus_labels_tr, nordeus_val_prepared, nordeus_labels_vl):
    gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120)
    gbrt.fit(nordeus_train_prepared, nordeus_labels_tr)

    errors = [mean_squared_error(nordeus_labels_vl, y_pred)
              for y_pred in gbrt.staged_predict(nordeus_val_prepared)]

    bst_n_estimators = np.argmin(errors) + 1
    gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators)
    gbrt_best.fit(nordeus_train_prepared, nordeus_labels_tr)

    print(errors)
    return gbrt_best
