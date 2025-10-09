from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import io
import logging
import traceback
import os

app = FastAPI()

# 🔓 Enable CORS for frontend/mobile integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 📁 Resolve absolute paths for model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(BASE_DIR, "imputer.pkl"), "rb") as f:
        imputer = pickle.load(f)

    with open(os.path.join(BASE_DIR, "feature_columns.pkl"), "rb") as f:
        feature_columns = pickle.load(f)

except FileNotFoundError as fnf_error:
    logging.error(f"Missing model file: {fnf_error}")
    raise RuntimeError("One or more model files are missing. Please check deployment artifacts.")
except Exception as e:
    logging.error(f"Model loading failed:\n{traceback.format_exc()}")
    raise RuntimeError(f"Failed to load model artifacts due to unexpected error: {e}")

# 🩺 Health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}

# 🔮 Prediction route
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        # 🧾 Validate file format
        if not file.filename.endswith((".xlsx", ".xls")):
            return {"error": "Invalid file format. Please upload an Excel file."}

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
        logging.error(f"Prediction failed:\n{traceback.format_exc()}")
        return {"error": f"Prediction failed: {str(e)}"}