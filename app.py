import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('potatoes.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
IMAGE_SIZE = 224

# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Streamlit UI
st.title("Potato Disease Detection")
st.write("Upload an image of a potato leaf and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Open the image using PIL
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)  # Updated parameter
    
    # Resize for model
    img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    
    # Predict
    predicted_class, confidence = predict(img_resized)
    
    # Display result
    st.success(f"Predicted Class: {predicted_class}")
    st.info(f"Confidence: {confidence}%")
