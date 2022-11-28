from sklearn.svm import SVR


def support(X, y):
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    return svm_poly_reg.fit(X, y)