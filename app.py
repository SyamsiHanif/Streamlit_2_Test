import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# Function to load the model
def load_model():
    try:
        model = joblib.load('random_forest_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the trained Random Forest model
model = load_model()

# Frequency Encoding Function
def frequency_encoding(df, columns):
    for col in columns:
        frequency = df[col].value_counts()
        df[f'{col}_freq'] = df[col].map(frequency)
    return df

# Label Encoding Function
def label_encoding(df, columns):
    le = LabelEncoder()
    for col in columns:
        df[f'{col}_encoded'] = le.fit_transform(df[col])
    return df

# Function to predict price using the trained model
def predict_price(data):
    if model is None:
        st.error("Model is not loaded. Please check the model file.")
        return None
    
    # Apply Frequency Encoding
    freq_columns = ['Manufacturer', 'Model', 'Category', 'Fuel type', 'Color']
    data = frequency_encoding(data, freq_columns)
    
    # Apply Label Encoding
    label_columns = ['Leather interior', 'Gear box type', 'Drive wheels', 'Doors', 'Wheel']
    data = label_encoding(data, label_columns)
    
    # Define feature columns
    features = [
        'Prod. year', 'Engine volume', 'Mileage', 'Cylinders', 'Airbags',
        'Manufacturer_freq', 'Model_freq', 'Category_freq', 'Fuel type_freq', 'Color_freq',
        'Leather interior_encoded', 'Gear box type_encoded', 'Drive wheels_encoded',
        'Doors_encoded', 'Wheel_encoded'
    ]
    
    # Ensure all feature columns are present in the data
    for feature in features:
        if feature not in data.columns:
            data[feature] = 0

    # Predict using the Random Forest model
    try:
        prediction = model.predict(data[features])
        return prediction[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Streamlit form
st.title("Car Price Prediction")

with st.form(key='car_form'):
    prod_year = st.selectbox('Production Year', [2010, 2011, 2006, 2014, 2016, 2013, 2007, 1999, 1997, 2018, 2008, 2012, 2017, 2001,
              1995, 2009, 2000, 2019, 2015, 2004, 1998, 1990, 2005, 2003, 1985, 1996, 2002, 1993,
              1992, 1988, 1977, 1989, 1994, 2020, 1984, 1986, 1991, 1983, 1953, 1964, 1974, 1987,
              1943, 1978, 1965, 1976, 1957, 1980, 1939, 1968, 1947, 1982, 1981, 1973])
    engine_volume = st.number_input('Engine Volume (L)', min_value=0.1, value=2.0)
    mileage = st.number_input('Mileage (km)', min_value=0, value=50000)
    cylinders = st.selectbox('Number of Cylinders', [6, 4, 8, 1, 12, 3, 2, 16, 5, 7, 9, 10, 14])
    airbags = st.number_input('Number of Airbags', min_value=0, value=6)
    manufacturer = st.selectbox('Manufacturer', ['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 'MERCEDES-BENZ',
                  'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 'RENAULT', 'NISSAN',
                  'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 'MAZDA', 'GMC', 'FIAT',
                  'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA', 'LINCOLN', 'VAZ', 'GAZ', 'CITROEN',
                  'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU',
                  'BUICK', 'TESLA', 'CADILLAC', 'PEUGEOT', 'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL',
                  'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ', 'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH',
                  'MASERATI', 'FERRARI', 'SAAB', 'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC', 'SATURN',
                  'ASTON MARTIN', 'GREATWALL'])
    model_input = st.text_input('Model', value='Camry')
    category = st.selectbox('Category', ['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 'Universal', 'Coupe',
               'Minivan', 'Cabriolet', 'Limousine', 'Pickup'])
    fuel_type = st.selectbox('Fuel Type', ['Hybrid', 'Petrol', 'Diesel', 'CNG', 'Plug-in Hybrid', 'LPG', 'Hydrogen'])
    color = st.selectbox('Color', ['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 'Red', 'Sky blue', 'Orange',
          'Yellow', 'Brown', 'Golden', 'Beige', 'Carnelian red', 'Purple', 'Pink'])
    leather_interior = st.selectbox('Leather Interior', ['Yes', 'No'])
    gear_box_type = st.selectbox('Gear Box Type', ['Automatic', 'Tiptronic', 'Variator', 'Manual'])
    drive_wheels = st.selectbox('Drive Wheels', ['4x4', 'Front', 'Rear'])
    doors = st.selectbox('Number of Doors', ['2-3', '4-5', '>5'])
    wheel = st.selectbox('Wheel', ['Left wheel', 'Right-hand drive'])

    submit_button = st.form_submit_button(label='Predict Price')

    if submit_button:
        new_data = {
            'Prod. year': [prod_year],
            'Engine volume': [engine_volume],
            'Mileage': [mileage],
            'Cylinders': [cylinders],
            'Airbags': [airbags],
            'Manufacturer': [manufacturer],
            'Model': [model_input],
            'Category': [category],
            'Fuel type': [fuel_type],
            'Color': [color],
            'Leather interior': [leather_interior],
            'Gear box type': [gear_box_type],
            'Drive wheels': [drive_wheels],
            'Doors': [doors],
            'Wheel': [wheel]
        }

        new_df = pd.DataFrame(new_data)
        predicted_price = predict_price(new_df)
        
        if predicted_price is not None:
            st.write(f"Predicted Car Price: ${predicted_price:,.2f}")
