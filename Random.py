import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def randfor(nordeus_train_prepared, nordeus_labels_tr, nordeus_val_prepared, nordeus_labels_vl):
    randFOR = RandomForestRegressor(max_depth=2, n_estimators=120)
    randFOR.fit(nordeus_train_prepared, nordeus_labels_tr)

    return randFOR


