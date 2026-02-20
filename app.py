import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# 1. Define Input Schema using Pydantic
class AutoVestRequest(BaseModel):
    daily_spending: float = Field(..., description="Total spent today in dollars")
    spare_change_total: float = Field(..., description="Round-up amount available")
    spending_variance: float = Field(..., description="Variance in spending over last 7 days")
    emergency_balance_ratio: float = Field(..., description="Ratio of current emergency funds to target")
    market_risk_score: float = Field(..., description="Current market risk index (0.0 to 1.0)")
    user_type: str = Field(..., description="'student' or 'professional'")

# 2. Initialize FastAPI App
app = FastAPI(title="AutoVest Decision Engine", version="1.0.0")

# 3. Load Model safely on startup
model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load("autovest_model.pkl")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")

# 4. Prediction Endpoint
@app.post("/predict")
async def predict_investment(request: AutoVestRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model is not loaded.")
        
    try:
        # Convert request to DataFrame
        input_data = pd.DataFrame([request.dict()])
        
        # Make Prediction and get Probabilities
        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]
        confidence = round(float(np.max(probabilities)), 4)
        decision = "Invest" if prediction == 1 else "Pause"
        
        # --- Reasoning Engine ---
        # Calculate pseudo-local importance: (abs(Standardized Input) * Global Importance)
        preprocessor = model.named_steps['preprocessor']
        classifier = model.named_steps['classifier']
        
        # Transform the single input and get feature names
        transformed_input = preprocessor.transform(input_data).flatten()
        feature_names = preprocessor.get_feature_names_out()
        global_importances = classifier.feature_importances_
        
        # Calculate impact score for this specific user
        impact_scores = np.abs(transformed_input) * global_importances
        
        # Zip, sort, and extract top 3 contributing features
        feature_impacts = list(zip(feature_names, impact_scores))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        
        # Clean up feature names for the API response (e.g., 'num__emergency_balance_ratio' -> 'emergency_balance_ratio')
        top_3_features = [f[0].split('__')[-1] for f in feature_impacts[:3]]
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": top_3_features
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Run locally using: uvicorn app:app --reload