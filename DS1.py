import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# Load the pre-trained models
destination_model = joblib.load('destination_model.pkl')
restaurant_model = joblib.load('restaurant_model.pkl')
destination_data = pd.read_csv('C:/Users/Home/Documents/GitHub/Intelligent-Travel-Planner/destinations.csv') 

# Streamlit user input form
# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .stButton>button {
            background-color: #0078D4;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
        }
        .stButton>button:hover {
            background-color: #005fa3;
        }
        h1 {
            color: #0078D4;
            font-size: 40px;
            font-weight: bold;
        }
        h2 {
            color: #003366;
            font-size: 30px;
            font-weight: normal;
        }
    </style>
""", unsafe_allow_html=True)


st.title("🌍 Intelligent Travel Planner ✈️")

st.header("🧳 Enter your preferences 👇")

st.markdown("### Your preferred destination will be predicted based on these inputs! 🌴")


# User inputs for the new user
st.text("Enter your age: ")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
budget_range = st.selectbox("Budget Range", ['5000-10000', '10000-20000', '1000-5000'])

st.text("choose any climate that you want: ")
preferred_climate_cold = st.checkbox("Cold Climate")
preferred_climate_moderate = st.checkbox("Moderate Climate")
preferred_climate_sunny = st.checkbox("Sunny Climate")

st.text("choose activities that you like: ")
preferred_activities_adventure = st.checkbox("Adventure")
preferred_activities_hiking = st.checkbox("Hiking")
preferred_activities_photography = st.checkbox("Photography")
preferred_activities_sightseeing = st.checkbox("Sightseeing")
preferred_activities_trekking = st.checkbox("Trekking")
preferred_activities_wildlife = st.checkbox("Wildlife")

# Create DataFrame from user input
user_input = {
    'age': [age],
    'budget_range': [budget_range],
    'preferred_climate_Cold': [preferred_climate_cold],
    'preferred_climate_Moderate': [preferred_climate_moderate],
    'preferred_climate_Sunny': [preferred_climate_sunny],
    'preferred_activities_Adventure': [preferred_activities_adventure],
    'preferred_activities_Hiking': [preferred_activities_hiking],
    'preferred_activities_Photography': [preferred_activities_photography],
    'preferred_activities_Sightseeing': [preferred_activities_sightseeing],
    'preferred_activities_Trekking': [preferred_activities_trekking],
    'preferred_activities_Wildlife': [preferred_activities_wildlife]
}

user_df = pd.DataFrame(user_input)

# # Create a mapping dictionary
destination_mapping = dict(zip(destination_data['destination_id'], destination_data['destination_name']))



# Predict destination and restaurant using the pre-trained models
predicted_destination = destination_model.predict(user_df)
predicted_restaurant = restaurant_model.predict(user_df)

# Extract the first predicted destination ID from the array
predicted_destination_id = predicted_destination[0]

# Get the destination name from the mapping dictionary
predicted_destination_name = destination_mapping.get(predicted_destination_id, "Unknown Destination")

# Display predictions
with st.expander("Prediction Results"):
    st.subheader(f"Predicted Destination ID: {predicted_destination_id}")
    # predicted_destination_name = destination_data[['destination_id' == predicted_destination]]['destination_name']
    st.subheader(f"Predicted Destination Name: {predicted_destination_name}")
    st.subheader(f"Predicted Restaurant: {predicted_restaurant[0]}")
    # predicted_destination_name = destination_data[destination_data['destination_id'] == predicted_destination]['destination_name']

    
# option of id:
# user_entered_id = st.number_input("Enter your Destination ID", min_value=1, max_value=500, step=1)

# # Display destination name and restaurant based on entered ID
# if user_entered_id:
#     # Manually add the entered destination_id to the user input
#     user_input_with_id = user_df.copy()
#     user_input_with_id['destination_id'] = user_entered_id  # Add the entered ID to the dataframe

    # Use the restaurant model to predict restaurant for entered ID
    # user_restaurant = restaurant_model.predict(user_input_with_id)

    # # Get the destination name from the mapping
    # user_destination_name = destination_mapping.get(user_entered_id, "Unknown Destination")

    # # Display results
    # st.subheader(f"Destination Name: {user_destination_name}")
    # st.subheader(f"Famous Restaurant: {user_restaurant[0]}")

submit_button = st.button("Submit", key="submit", help="Click to get predictions!")
if submit_button:
    st.write("Your predictions are ready! 🎉")
