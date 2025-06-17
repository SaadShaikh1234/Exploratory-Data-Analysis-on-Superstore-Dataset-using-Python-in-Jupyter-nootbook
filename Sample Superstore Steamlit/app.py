import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

# Load the trained Random Forest model
model_filename = 'random_forest_model.pkl'
try:
    with open(model_filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error(
        f"Error: Model file '{model_filename}' not found.  Place it in the same directory as this app.py file."
    )
    st.stop()

# Load the original dataset (for encoding categories - important for consistency)
DATA_PATH = 'SampleSuperstore.csv'
try:
    original_store = pd.read_csv(DATA_PATH, encoding='Latin1')
except FileNotFoundError:
    st.error(
        f"Error: Data file '{DATA_PATH}' not found. Place it in the same directory as this app.py file."
    )
    st.stop()

# Identify categorical columns
categorical_cols = original_store.select_dtypes(include='object').columns.tolist()
if 'Profit' in categorical_cols:
    categorical_cols.remove('Profit')  # Don't encode the target here

# Clean and standardize the categorical columns in the original data
for col in categorical_cols:
    original_store[col] = original_store[col].astype(str).str.lower().str.strip()

# Initialize LabelEncoders and fit them with the original data
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle errors during fitting.
    try:
        original_store[col] = le.fit_transform(original_store[col])
    except Exception as e:
        st.error(f"Error fitting LabelEncoder for column '{col}': {e}")
        st.stop()
    label_encoders[col] = le

# Get all unique values from the original dataframe.
original_unique_values = {}
for col in categorical_cols:
    original_unique_values[col] = original_store[col].unique().tolist()


# Define the prediction function
def predict_profit_loss(data):
    df = pd.DataFrame([data])

    # Encode categorical features using the fitted LabelEncoders
    for col, le in label_encoders.items():
        if col in df.columns:
            try:
                # Convert input to lowercase and strip whitespace for consistency
                df[col] = df[col].astype(str).str.lower().str.strip()
                df[col] = le.transform(df[col])
            except ValueError as e:
                st.warning(
                    f"Warning: Unseen label '{df[col].iloc[0]}' in column '{col}'.  "
                    f"Prediction may be unreliable.  Consider checking your input."
                )
                return None

    # Ensure all necessary columns are present (handle potential missing columns)
    feature_cols = [
        'Ship Mode',
        'Segment',
        'Country',
        'City',
        'State',
        'Postal Code',
        'Region',
        'Category',
        'Sub-Category',
        'Sales',
        'Quantity',
        'Discount',
    ]
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0  # Or some other appropriate default value,  0 is common for missing

    # Select and order the features as they were during training
    df = df[feature_cols]

    # Perform scaling (if your pipeline includes it)
    try:
        if hasattr(model.named_steps['std_scaler'], 'transform'):  # Check if scaler exists
            scaled_data = model.named_steps['std_scaler'].transform(df)
            prediction = model.named_steps['rfc_classifier'].predict(scaled_data)
        else:
            prediction = model.named_steps['rfc_classifier'].predict(df)
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

    return prediction[0]


# Streamlit app
def main():
    # Set the title of the web page
    st.title("Superstore Profit/Loss Prediction")

    # Add some descriptive text
    st.write("Enter the order details to predict if it will result in a Gain or Loss.")

    # Define the input fields for the user to enter data
    # Use text_input instead of selectbox
    ship_mode = st.text_input("Ship Mode", "")
    segment = st.text_input("Segment", "")
    country = st.text_input("Country", "")
    city = st.text_input("City", "")
    state = st.text_input("State", "")
    # Handle empty Postal Code case.
    postal_code_min = (
        int(original_store['Postal Code'].min())
        if not original_store['Postal Code'].empty
        else 0
    )
    postal_code_max = (
        int(original_store['Postal Code'].max())
        if not original_store['Postal Code'].empty
        else 0
    )
    postal_code_mean = (
        int(original_store['Postal Code'].mean())
        if not original_store['Postal Code'].empty
        else 0
    )
    postal_code = st.number_input(
        "Postal Code",
        min_value=postal_code_min,
        max_value=postal_code_max,
        value=postal_code_mean,
    )
    region = st.text_input("Region", "")
    category = st.text_input("Category", "")
    sub_category = st.text_input("Sub-Category", "")
    sales = st.number_input("Sales", min_value=0.0)
    quantity = st.number_input("Quantity", min_value=1, step=1)
    discount = st.slider("Discount", min_value=0.0, max_value=1.0, step=0.01)

    result = ""  # Initialize result

    # When the user clicks the 'Predict' button
    if st.button("Predict"):
        # Get the input data from the user
        input_data = {
            'Ship Mode': ship_mode,
            'Segment': segment,
            'Country': country,
            'City': city,
            'State': state,
            'Postal Code': postal_code,
            'Region': region,
            'Category': category,
            'Sub-Category': sub_category,
            'Sales': sales,
            'Quantity': quantity,
            'Discount': discount,
        }

        # Make the prediction
        prediction_result = predict_profit_loss(input_data)

        # Display the result
        st.subheader("Prediction:")
        if prediction_result is not None:
            if prediction_result == 0:  # Assuming 0 for Gain, 1 for Loss
                st.success("This order is likely to result in a Gain.")
            elif prediction_result == 1:
                st.error("This order is likely to result in a Loss.")
            else:
                st.warning(
                    "Prediction result is uncertain."
                )  # Should not happen with binary
        else:
            st.warning(
                "Could not make a reliable prediction due to unseen input data.  Please check your inputs."
            )


if __name__ == '__main__':
    main()
