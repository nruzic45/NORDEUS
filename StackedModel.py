from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LinearRegression


def stacked(X, y, estimators):
    # Build stack model

    lr = LinearRegression(solver='liblinear', random_state=0).fit(X, y)

    for i, estimator in enumerate(estimators):
        estimator_list = [
            ('i', estimator)]

    estimator_list.append(('lr', lr))

    stack_model = StackingClassifier(
        estimators=estimator_list,
        final_estimator=LinearRegression)

    # Train stacked model
    stack_model.fit(X, y)

    return stack_model
