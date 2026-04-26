from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()

# 🔹 Load models
reg_model = joblib.load("reg_model.pkl")
clf_model = joblib.load("clf_model.pkl")
columns = joblib.load("columns.pkl")


# 🔹 Prediction function
def predict_footprint(input_data):
    df = pd.DataFrame([input_data])
    
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    footprint = reg_model.predict(df)[0]
    impact = clf_model.predict(df)[0]

    return float(footprint), str(impact)


# 🔹 Suggestion system (your smart logic)
def suggest(data):
    suggestions = []

    if data['transport_mode'] == 'Car':
        if data['distance_km'] > 10:
            suggestions.append("🚗 High car usage → Use public transport or carpool")
        else:
            suggestions.append("🚗 Avoid short car trips, walk or bike instead")

    if data['electricity_kwh'] > 6:
        suggestions.append("💡 High electricity usage → Reduce AC & appliances")

    if data['renewable_usage_pct'] < 30:
        suggestions.append("🌞 Increase renewable energy usage")

    if data['food_type'] == 'Non-Veg':
        suggestions.append("🥗 Reduce non-veg meals")

    if data['waste_generated_kg'] > 0.7:
        suggestions.append("♻️ Reduce waste and recycle")

    return suggestions


# 🔹 API Endpoint
@app.get("/")
def home():
    return {"message": "Carbon AI API running 🚀"}
@app.post("/predict")
def predict(data: dict):
    footprint, impact = predict_footprint(data)
    suggestions = suggest(data)

    return {
    "carbon_footprint": round(float(footprint), 2),
    "impact": impact,
    "suggestions": suggestions
    }
import os

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
