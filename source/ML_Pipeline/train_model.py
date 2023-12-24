# Gerekli kütüphaneleri ve fonksiyonları içe aktar
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from ML_Pipeline.utils import max_val_index
from ML_Pipeline.model_evaluation import evaluate_model
import warnings

# Basitlik için uyarıları görmezden gel
warnings.simplefilter(action='ignore')

# Modelleri eğitmek için bir fonksiyon tanımla
def train_model(X_train, y_train, X_test, y_test, method):
    # Farklı sınıflandırıcıları ve varsayılan parametre değerlerini içeren bir sözlük
    model_dict = {
        'logistic_reg': LogisticRegression(solver="liblinear"),
        'naive_bayes': GaussianNB(),
        'svm_model': svm.SVC(gamma=0.25, C=10),
        'decision_tree': DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1),
        'rfcl': RandomForestClassifier(random_state=1),
        'adaboost': AdaBoostClassifier(),
        'xgboost': XGBClassifier(),
        'knn': KNeighborsClassifier(n_neighbors=3),
    }

    # Uyumlu modelleri ve karşılık gelen skorları depolamak için listeler
    fitted_models = []
    scores = []

    # Sözlükteki her model üzerinde döngü yap
    for model_name, model in model_dict.items():
        # Modeli eğitim verileri üzerinde eğit
        fitted_models.append(model.fit(X_train, y_train))
        
        # Modeli, verilen değerlendirme yöntemini kullanarak değerlendir
        scores.append(evaluate_model(y_test, model.predict(X_test), method))

    # Maksimum skorun indeksini ve değerini bul
    max_test = max_val_index(scores)
    
    # Maksimum skoru ve karşılık gelen model indeksini çıkart
    max_score = max_test[0]
    max_score_index = max_test[1]
    
    # Maksimum skora dayanarak final modeli seç
    final_model = fitted_models[max_score_index]

    # Final modeli ve maksimum skorunu döndür
    return final_model, max_score
