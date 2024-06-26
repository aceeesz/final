import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, UnidentifiedImageError

model = load_model('finaltrain.h5')
class_names = ['Rain', 'Shine', 'Cloudy', 'Sunrise']

def preprocess_image(image, target_size=(40, 60)):
    image = image.resize(target_size)
    image = image.convert('L')  
    image = np.array(image)
    image = image / 255.0 
    image = image.flatten()  
    image = np.expand_dims(image, axis=0) 
    return image

def predict(image):
    p_image = preprocess_image(image)
    return model.predict(p_image)

st.title("Final Exam: Model Deployment in the Cloud")
uploaded_file = st.file_uploader("Upload a weather image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict(image)
        probabilities = prediction[0]  
        predicted_classes = [class_names[i] for i in range(len(class_names))]
        predicted_results = dict(zip(predicted_classes, probabilities))

        for class_name, probability in predicted_results.items():
            st.write(f"{class_name}: {probability}")
            
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class = class_names[predicted_class_index]

        st.success(f"Prediction: {predicted_class}")

    except UnidentifiedImageError:
        st.error("The uploaded file could not be identified as an image. Please upload a valid image file.")
 
st.markdown("""
### Instructions:
1. Upload the weather image (jpg, png, jpeg).
2. Wait for the prediction to be displayed.
""")
