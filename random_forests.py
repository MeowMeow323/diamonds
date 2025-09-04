import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Load data
df = pd.read_csv("diamonds.csv")

# Drop 'Unnamed: 0' if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(['Unnamed: 0'], axis=1)

# Remove rows where x, y, or z are zero
df = df[(df['x'] != 0) & (df['y'] != 0) & (df['z'] != 0)]

# Create volume feature
df['volume'] = df['x'] * df['y'] * df['z']

# Cap outliers in volume, depth, table using IQR
for col in ['volume', 'depth', 'table']:
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_whisker = q1 - 1.5 * iqr
    upper_whisker = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=lower_whisker, upper=upper_whisker)

# Label encode cut, color, clarity
le_cut = LabelEncoder()
le_color = LabelEncoder()
le_clarity = LabelEncoder()
df['cut'] = le_cut.fit_transform(df['cut'])
df['color'] = le_color.fit_transform(df['color'])
df['clarity'] = le_clarity.fit_transform(df['clarity'])

# Features and target
X = df.drop(['price'], axis=1)
y = df['price']

# Train/test split (use 30% test as in notebook)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

# MinMaxScaler normalization
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=1)
rf.fit(X_train_scaled, y_train)

# Save model and encoders/scaler for app use
joblib.dump(rf, "rf_model.joblib")
joblib.dump(le_cut, "le_cut.joblib")
joblib.dump(le_color, "le_color.joblib")
joblib.dump(le_clarity, "le_clarity.joblib")
joblib.dump(scaler, "scaler.joblib")