from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import io
import logging

app = FastAPI()

# 🔓 Enable CORS for frontend/mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🧠 Load model artifacts
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("imputer.pkl", "rb") as f:
        imputer = pickle.load(f)

    with open("feature_columns.pkl", "rb") as f:
        feature_columns = pickle.load(f)

except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# 🩺 Health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}

# 🔮 Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))

        # ✅ Validate required columns
        missing = [col for col in feature_columns if col not in df.columns]
        if missing:
            return {"error": f"Missing columns: {missing}"}

        # 🧼 Preprocess
        df = df[feature_columns]
        df_imputed = pd.DataFrame(imputer.transform(df), columns=feature_columns)

        # 🔮 Predict
        predictions = model.predict_proba(df_imputed)[:, 1]
        risk_labels = ["High Risk" if p > 0.7 else "Low Risk" for p in predictions]

        # 📊 Return results
        result_df = df.copy()
        result_df["Autism Probability"] = predictions
        result_df["Risk Level"] = risk_labels

        logging.info(f"Processed file: {file.filename} with {len(df)} rows")
        return result_df.to_dict(orient="records")

    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        return {"error": str(e)}