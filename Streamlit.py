import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("cnn_celeb.keras")

# Function to make predictions
def make_prediction(img, model):
    img = Image.open(img)
    img = img.resize((128, 128))
    img = np.array(img)
    img = img / 255.0
    input_img = np.expand_dims(img, axis=0)
    res = model.predict(input_img,verbose=0)

    # Get the predicted class
    predicted_class = np.argmax(res)

    # Define class labels
    class_labels = ["Messi", "Sharapova", "Federer", "Williams", "Kohli"]

    # Display the prediction
    st.write(f"Prediction: {class_labels[predicted_class]}")

# Streamlit app
st.title("Athlete Image Classifier")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Make prediction when an image is uploaded
if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=False,width=300)

    # Make prediction button
    if st.button("Make Prediction"):
        make_prediction(uploaded_file, model)
