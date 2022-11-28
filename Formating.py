import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()

def solution(model, predict_setX):
    predict_sety = model.predict(predict_setX)
    revertedy = sc_X.inverse_transform(predict_sety)
    revertedX = sc_X.inverse_transform(predict_setX)
    return pd.to_csv(np.c_[revertedX, revertedy])
