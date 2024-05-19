# -*- coding: utf-8 -*-
"""
Created on Sun May 19 15:11:46 2024

@author: Computer
"""
import numpy as np
import streamlit as st
from PIL import Image
from keras.models import load_model

st.title("Finals Exam: Cloud Deployment")

model = load_model('finaltrain.h5')
file = st.file_uploader("Upload File Here", type = ["png","jpeg","jpg"])
classification = {
    0: "Cloudy", 1: "Rain", 2: "Sunrise", 3:"Shine"}
if file is not None: 
    st.image(file)
    image = Image.open(file)
    image = image.resize((224, 224))
    image = np.expand_dims(image, axis=0)
    image = np.array(image) / 255.0
        
    predict = model.predict(image)
    predict_class = np.argmax(predict, axis=1)[0]
    
    
    if predict_class in classification:
        predict = classification[predict_class]
        st.success(f"Prediction: {predict}")
    else:
        st.warning("Cannot Predict")
    
