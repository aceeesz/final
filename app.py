import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

# Load the trained model
model = load_model('finaltrain.h5')
class_names=['Rain','Shine','Cloudy','Sunrise']
# Function to preprocess the input image
def preprocess_image(image, target_size=(60, 40)):
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make predictions
def predict(image):
    p_image = preprocess_image(image)
    return model.predict(p_image)

# Streamlit app
st.title("Final Exam: Model Deployment in the Cloud")

# Upload image
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    
    prediction = model.predict(image)
    predicted_samp = np.argmax(prediction, axis=1)[0]
    
    if predict_class in classification:
        predict = classification[predict_class]
        st.success(f"Prediction: {predict}")
    else:
        st.warning("Cannot Predict")

# Instructions for the user
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
3. Click "Calculate Accuracy" to see how well the model predicts the temperature.
""")
