# app/api.py
from fastapi import FastAPI, UploadFile, File
from src.utils.prediction_utils import load_image_from_upload
from src.pipelines.predict import prediction_pipeline

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Welcome to the Alpaca Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = load_image_from_upload(file)
    prediction = prediction_pipeline(image)
    return {"prediction": prediction}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, port=8000)
