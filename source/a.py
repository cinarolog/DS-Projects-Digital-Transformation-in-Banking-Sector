import streamlit as st
import pickle
from scipy.stats import zscore

# Gerekli fonksiyonları içe aktar
from sklearn.model_selection import train_test_split
from ML_Pipeline.utils import read_data, merge_dataset, drop_col, null_values
from ML_Pipeline.train_model import train_model
from ML_Pipeline.grid_model import grid_model
from ML_Pipeline.hyper_parameters import choose_param_grid
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from collections import Counter

# Veriyi oku
data1 = read_data("../input/Data1.csv")
data2 = read_data("../input/Data2.csv")

# Veriyi birleştir
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

# Modeli dengesiz veri üzerinde eğit
XScaled = final_data.drop(['LoanOnCard'], axis=1).apply(zscore)
counter = Counter(final_data['LoanOnCard'])
over = SMOTE(sampling_strategy=0.3, random_state=1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipeline = Pipeline(steps=steps)
Xb, Yb = pipeline.fit_resample(XScaled, final_data['LoanOnCard'])
counter = Counter(Yb)
x_trainb, x_testb, y_trainb, y_testb = train_test_split(Xb, Yb, test_size=0.3, random_state=1)
param_grid = choose_param_grid("dt_param_grid")
model_grid_search = grid_model(x_trainb, y_trainb, param_grid, 'decision_tree')
score_method = "accuracy_score"
final_model, max_score = train_model(x_trainb, y_trainb, x_testb, y_testb, score_method)

# Eğitilmiş modeli dosyadan yükle
loaded_model = pickle.load(open('../output/finalized_model.sav', 'rb'))

# Streamlit uygulaması
st.title('Loan On Card Prediction App')

# Kullanıcıdan girişleri al
user_input = st.text_input('Please enter the features for prediction (comma-separated):')

if user_input:
    # Kullanıcının girdisini işle
    features = list(map(float, user_input.split(',')))
    
    # Modelin tahmin yapması
    prediction = loaded_model.predict([features])

    # Tahmin sonucunu göster
    st.write('Prediction:', prediction[0])
