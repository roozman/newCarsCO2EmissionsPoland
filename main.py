import uvicorn
from pydantic import BaseModel, Field
from fastapi import FastAPI
from predict import predict

app = FastAPI()

class InputData (BaseModel):
    m_kg: float = Field(..., alias="m_(kg)")
    mt: float
    ec_cm3: float = Field(..., alias="ec_(cm3)")
    ep_kw: float = Field(..., alias="ep_(kw)")
    fuel_consumption_: float
    mk: str
    ech: str
    ft: str
    fm: str

@app.post("/predict")
async def get_predictions(df: InputData):
    input_data = df.model_dump(by_alias=True)
    predictions = predict(input_data)
    return {'predictions': predictions}

@app.get('/')
async def root():
    return{"This is the root directory of my MLZoomcomp2025 midterm project, named CO2emissionprediction"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)