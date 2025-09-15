import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
import joblib

df = pd.read_csv("../data/cleaned.csv")
df = df.drop(columns=['MarketEstimate'])

X_raw = df[['Zipcode', 'Bedroom', 'Bathroom', 'Area', 'PPSq', 'ListedPrice']]
y = df['RentEstimate']

ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
ct = ColumnTransformer(transformers=[
    ('zip_ohe', ohe, ['Zipcode'])
], remainder='passthrough')  # passthrough the other numeric columns

X_encoded = ct.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)

y_pred = ridge.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"R^2 Score: {r2}")
print(f"RMSE: {rmse}")

joblib.dump(ridge, '../models/ridge_model.pkl')
joblib.dump(ct, '../models/column_transformer.pkl')