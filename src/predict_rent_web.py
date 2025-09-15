import joblib
import pandas as pd

# Load model and transformer
ridge = joblib.load('../models/ridge_model.pkl')
ct = joblib.load('../models/column_transformer.pkl')

def estimate_rent(
    zipcode: str, 
    bedrooms: float, 
    bathrooms: float, 
    unit_sqft: float, 
    listed_price: float, 
    building_sqft: float
) -> float:
    
    if unit_sqft != building_sqft: # We have a multi-family home.
        listed_price = (unit_sqft / building_sqft) * listed_price
    
    ppsq = listed_price / unit_sqft
    
    data = [[zipcode, bedrooms, bathrooms, unit_sqft, ppsq, listed_price]]
    df = pd.DataFrame(data, columns=["Zipcode", "Bedroom", "Bathroom", "Area", "PPSq", "ListedPrice"])
    
    encoded = ct.transform(df)
    predicted = ridge.predict(encoded)
    return float(predicted[0])