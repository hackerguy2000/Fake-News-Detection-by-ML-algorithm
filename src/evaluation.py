from sklearn.metrics import accuracy_score

def evaluate_model(Y_true, Y_pred):
    accuracy = accuracy_score(Y_true, Y_pred)
    return accuracy
