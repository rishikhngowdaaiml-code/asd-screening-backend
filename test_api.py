import requests
import pandas as pd

# 🔗 API endpoint
API_URL = "https://asd-screening-api.onrender.com/predict"

# 📂 Input Excel file
INPUT_FILE = "Asd-Child-Data.xlsx"
OUTPUT_FILE = "asd_predictions.xlsx"

try:
    # 🚀 Send Excel file to FastAPI backend
    with open(INPUT_FILE, "rb") as f:
        response = requests.post(API_URL, files={"file": f})

    # ❌ Check for request failure
    if response.status_code != 200:
        raise Exception(f"API returned status code {response.status_code}: {response.text}")

    # 📊 Convert response to DataFrame
    data = response.json()
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("API response is empty or malformed")

    results = pd.DataFrame(data)

    # 💾 Save predictions locally
    results.to_excel(OUTPUT_FILE, index=False)
    print(f"✅ Saved predictions to {OUTPUT_FILE}")

    # 🔍 Show sample predictions
    print("\n🔍 Sample Predictions:")
    print(results.head(10).to_string(index=False))

    # 🚨 Show high-risk cases
    high_risk = results[results["Risk Level"] == "High Risk"]
    print("\n🚨 High Risk Cases:")
    print(high_risk.to_string(index=False))

except Exception as e:
    print(f"\n❌ Error: {e}")