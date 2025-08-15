Potato Disease Detection using CNN and LSTM
Project Overview

This project implements a deep learning-based solution to detect diseases in potato leaves. The model combines Convolutional Neural Networks (CNN) for feature extraction and LSTM (Long Short-Term Memory) layers for sequential learning, providing accurate classification of potato leaf images.

Users can upload an image of a potato leaf through a Streamlit web interface, and the system will classify it into one of the following categories:

Potato___Early_blight

Potato___Late_blight

Potato___Healthy

Along with the predicted class, the model also provides a confidence score indicating the probability of the prediction.

Features

Deep Learning Model: Combines CNN and LSTM for better feature learning.

Interactive Web App: Built with Streamlit for easy upload and instant predictions.

Confidence Score: Provides a percentage confidence for the predicted class.

Multi-class Classification: Detects Early Blight, Late Blight, or Healthy leaves.

How to Use

Clone the repository.

Install required dependencies:

pip install -r requirements.txt


Run the Streamlit application:

streamlit run app.py


Upload an image of a potato leaf and get the predicted disease class along with the confidence score.

Output

When a potato leaf image is uploaded, the app displays:

The uploaded image.

Predicted Class: Early Blight, Late Blight, or Healthy.

Confidence Score: Probability of the predicted class in percentage.

Technologies Used

Python

TensorFlow / Keras

CNN & LSTM

Streamlit 
