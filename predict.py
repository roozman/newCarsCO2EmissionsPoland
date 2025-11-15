import joblib
import pandas as pd

pipe = joblib.load('model.bin')

def predict(df):
    df = pd.DataFrame([df])
    return {'predictions' : float(pipe.predict(df)[0])}