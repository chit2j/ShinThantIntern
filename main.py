#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import asyncio
import streamlit as st
import numpy as np
from keras.models import model_from_json


# In[2]:


#load face
try:
    face_cascade = cv2.CascadeClassifier('cascade/haarcascade_frontalface_default.xml')
except Exception:
    st.write("Error loading cascade classifiers")


# In[3]:


# Define the labels for the emotions
emotions= {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}


# In[4]:


# Load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)


# In[5]:


# Load into new model
model.load_weights('model/emotion_model.h5')
print("Loaded model from disk")


# In[6]:


# Function to detect faces and emotions in an image
def detect_faces_and_emotions(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x,y-50), (x+w, y+h+10), (0,255,0), 6)
        roi_frame = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_frame,(48, 48)), -1), 0)     

        # Predict the emotion of the face using the pre-trained model
        emotion_prediction = model.predict(cropped_img)
        max_index = int(np.argmax(emotion_prediction))
        cv2.putText(image, emotions[max_index], (x+5, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4, cv2.LINE_AA)
    return image


# In[7]:

# Streamlit app
st.title("Face  &  Emotion Detection")

st.sidebar.caption("Upload an image to detected!")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type="jpg")


if uploaded_file is not None:
    image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    st.write("Original Image")
    st.image(image, channels="BGR")

    detected_image = detect_faces_and_emotions(image)

    st.write("Detected Image")
    st.image(detected_image, channels="BGR")

else:
    pass


# In[ ]:




