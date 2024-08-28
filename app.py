import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

# add background to the app

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("https://cdn.pixabay.com/photo/2018/06/28/22/14/car-3504910_1280.jpg");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



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
st.title("Car Price Prediction 🚗")

with st.form(key='car_form'):
    # Changed to number inputs
    prod_year = st.number_input('Production Year', min_value=1900, max_value=2024, value=2020, step=1)
    engine_volume = st.number_input('Engine Volume (L)', min_value=0.1, value=2.0, step=0.1)
    mileage = st.number_input('Mileage (km)', min_value=0, value=50000, step=1000)
    cylinders = st.number_input('Number of Cylinders', min_value=1, max_value=16, value=4, step=1)
    airbags = st.number_input('Number of Airbags', min_value=0, value=6, step=1)
    
    # Sorting selection options alphabetically
    manufacturer = st.selectbox('Manufacturer', sorted(['LEXUS', 'CHEVROLET', 'HONDA', 'FORD', 'HYUNDAI', 'TOYOTA', 
                  'MERCEDES-BENZ', 'OPEL', 'PORSCHE', 'BMW', 'JEEP', 'VOLKSWAGEN', 'AUDI', 
                  'RENAULT', 'NISSAN', 'SUBARU', 'DAEWOO', 'KIA', 'MITSUBISHI', 'SSANGYONG', 
                  'MAZDA', 'GMC', 'FIAT', 'INFINITI', 'ALFA ROMEO', 'SUZUKI', 'ACURA', 
                  'LINCOLN', 'VAZ', 'GAZ', 'CITROEN', 'LAND ROVER', 'MINI', 'DODGE', 'CHRYSLER', 
                  'JAGUAR', 'ISUZU', 'SKODA', 'DAIHATSU', 'BUICK', 'TESLA', 'CADILLAC', 'PEUGEOT', 
                  'BENTLEY', 'VOLVO', 'სხვა', 'HAVAL', 'HUMMER', 'SCION', 'UAZ', 'MERCURY', 'ZAZ', 
                  'ROVER', 'SEAT', 'LANCIA', 'MOSKVICH', 'MASERATI', 'FERRARI', 'SAAB', 
                  'LAMBORGHINI', 'ROLLS-ROYCE', 'PONTIAC', 'SATURN', 'ASTON MARTIN', 'GREATWALL']))
    
    model_input = st.text_input('Model', value='Camry')
    
    category = st.selectbox('Category', sorted(['Jeep', 'Hatchback', 'Sedan', 'Microbus', 'Goods wagon', 
               'Universal', 'Coupe', 'Minivan', 'Cabriolet', 'Limousine', 'Pickup']))
    
    fuel_type = st.selectbox('Fuel Type', sorted(['Hybrid', 'Petrol', 'Diesel', 'CNG', 
                  'Plug-in Hybrid', 'LPG', 'Hydrogen']))
    
    color = st.selectbox('Color', sorted(['Silver', 'Black', 'White', 'Grey', 'Blue', 'Green', 
          'Red', 'Sky blue', 'Orange', 'Yellow', 'Brown', 'Golden', 'Beige', 
          'Carnelian red', 'Purple', 'Pink']))
    
    leather_interior = st.radio(
        'Leather Interior',
        options=['Yes', 'No'],
        index=0  # Default selection (optional)
    )
    
    gear_box_type = st.selectbox('Gear Box Type', sorted(['Automatic', 'Tiptronic', 'Variator', 'Manual']))
    
    drive_wheels = st.selectbox('Drive Wheels', sorted(['4x4', 'Front', 'Rear']))
    
    doors = st.number_input('Number of Doors', min_value=2, max_value=6, value=4, step=1)
    
    wheel = st.selectbox('Wheel', sorted(['Left wheel', 'Right-hand drive']))

    submit_button = st.form_submit_button(label='Predict Price')

    if submit_button:
        # Process doors input into categorical data
        if doors <= 3:
            doors_category = '2-3'
        elif doors <= 5:
            doors_category = '4-5'
        else:
            doors_category = '>5'

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
            'Doors': [doors_category],
            'Wheel': [wheel]
        }

        new_df = pd.DataFrame(new_data)
        predicted_price = predict_price(new_df)
        
        # if predicted_price is not None:
        #     st.write(f"Predicted Car Price: ${predicted_price:,.2f}")

        if predicted_price is not None:
            st.markdown(
                f"""
                <div style="font-size: 36px; font-weight: bold; color: #FFFFFF; text-align: left;">
                    Predicted price: ${predicted_price:,.2f}
                </div>
                """,
                unsafe_allow_html=True
            )
