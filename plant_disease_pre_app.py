# to run the app, command is: streamlit run name.py
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model


# loading the model
model = load_model("C:/Users/gurki/Desktop/New folder/DataML/plant_disease_predictor/plant_disease.h5")
#names of the classes
CLASS_NAMES = ['Corn-Common_rust','Potato-Early_blight','Tomato-Bacterial_spot']

#Setting the title of the app
st.title('Plant Disease Detection')
st.markdown('Upload an image of the plant leaf from: Corn common rust, Potato early blight, Tomato bacterial spot')

# Uploading the plant image
plant_image = st.file_uploader('Choose an Image..', type="jpg")
submit = st.button('Predict')

# The click of the button
if submit:
    if plant_image is not None:
        
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR")
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)
        #Make Prediction
        Y_pred = model.predict(opencv_image)
        result = CLASS_NAMES[np.argmax(Y_pred)]

        st.title(str("The Plant is suffering from "+ result))
