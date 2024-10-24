import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Mapbox token (replace with your own)
MAPBOX_TOKEN = 'pk.eyJ1Ijoia2FhbGFrYXJpa2FsYW4iLCJhIjoiY20yYjl2dzdqMHAydjJ3c2ZjYng1d2Q3YyJ9._iDR65eR_qkzbtrrcRjqag'

# Load the dataset
dataset_path = r"C:\Users\Admin\Documents\ML lab\CAT-2\Taxi-fare Prediction\Tamil_nadu_taxi_trips_.csv"
data = pd.read_csv(dataset_path)

# Calculate fare per km for each vehicle type
data['Fare_per_km'] = data['Fare_INR'] / data['Distance_km']

# Prepare the feature set and target
features = [
    'Distance_km', 'No_of_Passengers', 'Travel_Time_hrs', 'Tips_INR', 
    'Tourist_Place_Nearby', 'Weather_Condition', 'Vehicle_Type'
]
X = data[features]
y = data['Fare_INR']

# One-hot encode categorical features
categorical_features = ['Tourist_Place_Nearby', 'Weather_Condition', 'Vehicle_Type']
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
best_model = None
best_mse = float('inf')
best_r2 = 0
best_model_name = ""

# Store feature importances for feature selection
feature_importances = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Calculate performance metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"{name}:")
    print(f"  R² Score: {r2:.4f}")
    print(f"  Mean Squared Error (MSE): {mse:.4f}\n")
    
    # Select the model with the lowest MSE
    if mse < best_mse:
        best_mse = mse
        best_r2 = r2
        best_model = model
        best_model_name = name
        
    # Get feature importances for tree-based models
    if hasattr(model, 'feature_importances_'):
        feature_importances[name] = model.feature_importances_

# Print the best model and its evaluation metrics
print(f"Best Model: {best_model_name}")
print(f"  R² Score: {best_r2:.4f}")
print(f"  Mean Squared Error (MSE): {best_mse:.4f}")

# Feature Selection: Using the best model's feature importances (if applicable)
if best_model_name in feature_importances:
    importances = feature_importances[best_model_name]
    feature_names = X_encoded.columns
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    selected_features = feature_importance_df.nlargest(5, 'Importance')  # Select top 5 features

    print("\nSelected Features based on Importance:")
    print(selected_features)

    # Filter the dataset to include only selected features for prediction
    X_encoded = X_encoded[selected_features['Feature'].tolist()]

# Function to calculate distance using Mapbox API
def get_distance(pickup, dropoff):
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{pickup}.json?access_token={MAPBOX_TOKEN}"
    pickup_resp = requests.get(url).json()
    
    url = f"https://api.mapbox.com/geocoding/v5/mapbox.places/{dropoff}.json?access_token={MAPBOX_TOKEN}"
    dropoff_resp = requests.get(url).json()
    
    # Extracting coordinates (longitude, latitude)
    pickup_coords = pickup_resp['features'][0]['center']
    dropoff_coords = dropoff_resp['features'][0]['center']
    
    # Use the Mapbox Directions API to get the distance
    directions_url = f"https://api.mapbox.com/directions/v5/mapbox/driving/{pickup_coords[0]},{pickup_coords[1]};{dropoff_coords[0]},{dropoff_coords[1]}?access_token={MAPBOX_TOKEN}"
    directions_resp = requests.get(directions_url).json()
    
    # Extract the distance in kilometers
    distance_meters = directions_resp['routes'][0]['distance']
    distance_km = distance_meters / 1000  # convert meters to km
    
    return distance_km

# Function to predict fare based on distance and vehicle type
def predict_fare(distance_km, vehicle_type, num_passengers, travel_time, tips):
    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'Distance_km': [distance_km], 
        'No_of_Passengers': [num_passengers], 
        'Travel_Time_hrs': [travel_time], 
        'Tips_INR': [tips],
        'Vehicle_Type': [vehicle_type]  # Include all categorical variables
    })
    
    # One-hot encode the input data
    input_data_encoded = pd.get_dummies(input_data, columns=categorical_features, drop_first=True)
    
    # Ensure all vehicle type columns are present (in case some are missing in input data)
    for col in X_encoded.columns:
        if col not in input_data_encoded.columns:
            input_data_encoded[col] = 0  # Add missing vehicle type columns with 0
    
    # Predict the fare using the best model
    predicted_fare = best_model.predict(input_data_encoded)
    return predicted_fare[0]

# Get input from the user
pickup = input("Enter pickup location: ")
dropoff = input("Enter dropoff location: ")
vehicle_type = input("Enter vehicle type (e.g., SUV, Hatchback, Auto): ")
num_passengers = int(input("Enter number of passengers: "))
travel_time = float(input("Enter travel time in hours: "))
tips = float(input("Enter tips (optional): "))

# Calculate distance using Mapbox API
distance_km = get_distance(pickup, dropoff)
print(f"Calculated distance: {distance_km:.2f} km")

# Predict the fare
predicted_fare = predict_fare(distance_km, vehicle_type, num_passengers, travel_time, tips)
print(f"Predicted fare for {distance_km:.2f} km with vehicle type {vehicle_type}: INR {predicted_fare:.2f}")
