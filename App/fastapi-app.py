from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

app = FastAPI()

# Eğitilmiş modeli yükle
loaded_model = pickle.load(open('../output/finalized_model.sav', 'rb'))
scaler = StandardScaler()

# Özellikleri alacak Pydantic modeli tanımla
class Item(BaseModel):
    features: list

# /predict endpoint'i
@app.post("/predict")
def predict(item: Item):
    try:
        # Gelen özellikleri bir NumPy dizisine dönüştür
        features = np.array(item.features).reshape(1, -1)
        
        # Özellikleri ölçekle
        scaled_features = scaler.fit_transform(features)
        
        # Modeli kullanarak tahmin yap
        prediction = loaded_model.predict(scaled_features)
        
        # Tahmin sonucunu döndür
        return {"prediction": int(prediction[0])}
    except Exception as e:
        # Hata durumunda HTTPException fırlat
        raise HTTPException(status_code=500, detail=str(e))


#python -m uvicorn app:app --reload