from sklearn.linear_model import LogisticRegression

def train_model(X, Y):
    model = LogisticRegression()
    model.fit(X, Y)
    return model

def predict(model, X_new):
    Y_pred = model.predict(X_new)
    return Y_pred
