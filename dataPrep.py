import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from GBR import gbr
from NORDEUS.Random import randfor
from NORDEUS.StackedModel import stacked
from NORDEUS.Support import support
from stats import plot_learning_curves

data = pd.read_csv('jobfair.csv')

train_test_set = data.loc[data['date'] <= '2022-08-31']
predict_set = data.loc[data['date'] > '2022-08-31']

train_set, train_val_set = train_test_split(train_test_set, test_size=0.2, random_state=42)

test_size = 0.5

valid_set, test_set = train_test_split(train_val_set, test_size=0.5)


# print(traintest_set['device_model'].value_counts())
# print(data.describe())


# corr_matrix = train_test_set.corr()


#######################################################

# feature engineering class
# f1 - fresh users koji su dosli orgaically
# f2 - procenat korisnika registrovanih istog dana organically


class newAttribs(BaseEstimator, TransformerMixin):
    def __init__(self, add_fresh_and_organic=True):
        self.add_fresh_and_organic = add_fresh_and_organic

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X


####################################################### Preprocessing

nordeus = train_set.drop(["returned"], axis=1)
nordeus_labels_tr = train_set["returned"].copy()

nordeus_num = nordeus.drop(["date", "device_model", "os_version"], axis=1)
nordeus_cat = nordeus[["date", "device_model", "os_version"]]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', newAttribs()),
    ('std_scaler', StandardScaler()),
])

# Veoma mali broj vrednosti je NaN za model, probati sa frekvencijskim imputerom
cat_pipeline = Pipeline([
    ('encoderCat', OrdinalEncoder()),
    ('imputerCat', SimpleImputer()),

])

num_attribs = list(nordeus_num)
# Ordinal encoder, jer je u pitanju kodiranje intervala vremena
cat_attribs = ["date", "device_model", "os_version"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", cat_pipeline, cat_attribs),
])

nordeus_train_prepared = full_pipeline.fit_transform(nordeus)

nordeus_test = test_set.drop(["returned"], axis=1)
nordeus_labels_ts = test_set["returned"].copy()
nordeus_test_prepared = full_pipeline.fit_transform(nordeus_test)

nordeus_val = valid_set.drop(["returned"], axis=1)
nordeus_labels_vl = valid_set["returned"].copy()
nordeus_val_prepared = full_pipeline.fit_transform(nordeus_val)

# %% ############################models and evaluation
# param_grid = [
#     {'n_estimators': [10], 'max_features': [2, 4]},
#     {'bootstrap': [False], 'n_estimators': [10], 'max_features': [2]},
# ]
#
# grid_search = GridSearchCV(rand_forest, param_grid, cv=5,
#                            scoring='neg_mean_squared_error',
#                            return_train_score=True)
#
# grid_search.fit(nordeus_prepared, nordeus_labels)
# feature_importances = grid_search.best_estimator_.feature_importances_
# print(feature_importances)

#######################################################################################################
#
# vaznost = [0.07163234, 0.01509468, 0.01981881, 0.02002859, 0.01464922, 0.00222155,
#            0.0124963, 0.03201836, 0.0740941, 0.35478632, 0.28492889, 0.0729974,
#            0.02523345]
#
# attributes = num_attribs + cat_attribs
#
# vaznostAtributa = sorted(zip(vaznost, attributes), reverse=True)
# print(vaznostAtributa)

# [(0.35478632, 'registrations'), (0.28492889, 'date'), (0.0740941, 'device_memory_size_mb'), (0.0729974,
# 'device_model'), (0.07163234, 'registration_type'), (0.03201836, 'screen_dpi'), (0.02523345, 'os_version'),
# (0.02002859, 'network_type'), (0.01981881, 'registration_channel'), (0.01509468, 'played_t11_before'), (0.01464922,
# 'device_tier'), (0.0124963, 'device_manufacturer'), (0.00222155, 'device_type')]


# device_memory,tip registracije, datum i broj registracija u danu su najvazniji atributi.
# Os mi se cini da najmanje utice na povratak igraca, dok treba proveriti
# da li postoji specific relacija izmedju modela uredjaja i povratka(mozda odredjeni provajder koji ima bolji
# ugovor za protok podataka/taj model ima bolje specifikacije za igru)
# 2022-06-02 == najbolji prediktor povratka igraca(medju datumima)

# todo: proveri da li je veci procenat organic ili paid registracija u odredjenom danu -> nov feature

# todo: proveri dane kada je broj vracenih registrovanih prilicno veliki i vidi koji parametri odskacu od mean-a

########################################################################################################################

SVR = support(nordeus_train_prepared, nordeus_labels_tr)

gbr_best = gbr(nordeus_train_prepared, nordeus_labels_tr, nordeus_val_prepared, nordeus_labels_vl)

rf = randfor(nordeus_train_prepared, nordeus_labels_tr, nordeus_val_prepared, nordeus_labels_vl)

estimatorList = [SVR, gbr_best, rf]

stack = stacked(nordeus_train_prepared, nordeus_labels_tr, estimatorList)

plot_learning_curves(stack, nordeus_train_prepared, nordeus_labels_tr)
plt.show()
