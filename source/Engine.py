# Gerekli paketleri içe aktar
import pickle
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data, merge_dataset, drop_col, null_values
from ML_Pipeline.train_model import train_model
from ML_Pipeline.grid_model import grid_model
from ML_Pipeline.hyper_parameters import choose_param_grid
# imblearn paketlerini içe aktar
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter
from scipy.stats import zscore

# Veri setlerini oku
data1 = read_data("../input/Data1.csv")
data2 = read_data("../input/Data2.csv")

# Veri setlerini birleştir
final_data = merge_dataset(data1, data2, join_type='inner', on_param='ID')

# Gereksiz sütunları at
final_data = drop_col(final_data, ['ID', 'ZipCode', 'Age'])

# Eksik değerleri işle
final_data = null_values(final_data)

# Eğitim ve test verilerini oluştur
x_train, x_test, y_train, y_test = train_test_split(final_data.drop(['LoanOnCard'], axis=1),
                                                    final_data['LoanOnCard'],
                                                    test_size=0.3,
                                                    random_state=1)

# Modeli dengesiz veri üzerinde eğit - şu anlık devre dışı bırakılmış
# model = train_model(x_train, y_train, x_test, y_test)

# Veriyi ölçekle
XScaled = final_data.drop(['LoanOnCard'], axis=1).apply(zscore)

# LoanOnCard sınıf dengesizliğini kontrol et
counter = Counter(final_data['LoanOnCard'])

# SMOTE ve RandomUnderSampler kullanarak dengesiz veriyi dengele
over = SMOTE(sampling_strategy=0.3, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
Xb, Yb = pipeline.fit_resample(XScaled, final_data['LoanOnCard'])

# Dengelenmiş veriyi kontrol et
counter = Counter(Yb)

# Yeni eğitim ve test verilerini oluştur
x_trainb, x_testb, y_trainb, y_testb = train_test_split(Xb, Yb, test_size=0.3, random_state=1)

# SVM için hiperparametre grid'ini seç
param_grid = choose_param_grid("dt_param_grid")

# Grid arama ile SVM modelini eğit
model_grid_search = grid_model(x_trainb, y_trainb, param_grid, 'decision_tree')
print(model_grid_search)

# Skorlama yöntemini belirle
score_method = "accuracy_score"

# Dengelenmiş veri üzerinde eğitilmiş modeli değerlendir
final_model, max_score = train_model(x_trainb, y_trainb, x_testb, y_testb, score_method)
print(final_model)
print(max_score)

# Eğitilmiş modeli dosyaya kaydet
pickle.dump(final_model, open('../output/finalized_model.sav', 'wb'))
