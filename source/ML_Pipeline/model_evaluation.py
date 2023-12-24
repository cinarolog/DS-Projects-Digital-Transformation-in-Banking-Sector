from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, recall_score, f1_score, precision_score

"""
    Test ve tahmin datalarımızı verdikten sonra
    görmek istediğimiz score için method ismini veriyoruz,
    sonrası malum :)
"""
def evaluate_model(y_test, y_pred, method):
    if method == 'accuracy_score':
        score = accuracy_score(y_test, y_pred)
    elif method == 'precision_score':
        score = precision_score(y_test, y_pred)
    elif method == 'recall_score':
        score = recall_score(y_test, y_pred)
    elif method == 'f1_score':
        score = f1_score(y_test, y_pred)
    elif method == 'roc_auc_score':
        score = roc_auc_score(y_test, y_pred)
    else:
        print("Method kısmına bunlardan birini girmelisin =>'accuracy_score', 'precision_score', 'recall_score', 'f1_score' ve 'roc_auc_score'.")
        return None
    
    return score

def get_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    return cm