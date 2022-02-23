import tensorflow as tf
model = tf.keras.models.load_model("best.h5")
import streamlit as st
import cv2
import math
import numpy as np
import os
img_width=256; img_height=256
st.write("""
         # COVID-Pneumonia-Lung_Opacity-Normal Prediction
         """
         )
st.write("This is a simple image classification web app to predict COVID, Pneumonia, Lung Opacity and Normal")
file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
import cv2
from PIL import Image, ImageOps
import keras_ocr
from keras_ocr import tools
import numpy as np
def findout(pred):
    return pred
def preprocess_image(im):
    im.resize((256, 256, 3), refcheck=False)
    a = np.array(im)
    a = np.expand_dims(a, axis = 0)
    a =np.divide(a,255)
    return a
def midpoint(x1, y1, x2, y2):
    x_mid = int((x1 + x2)/2)
    y_mid = int((y1 + y2)/2)
    return (x_mid, y_mid)
def refine(prediction_groups,img):
    im=img
    for j in range(0,len(prediction_groups[0])):
        box = prediction_groups[0][j][1]
        x0, y0 = box[0]
        x1, y1 = box[1]
        x2, y2 = box[2]
        x3, y3 = box[3] 
        x_mid0, y_mid0 = midpoint(x1, y1, x2, y2)
        x_mid1, y_mi1 = midpoint(x0, y0, x3, y3)
        thickness = int(math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 ))
        g_mask = np.zeros(img.shape[:2],dtype='uint8')
        cv2.line(g_mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255, thickness)
        cv2.line(g_mask, (x_mid0, y_mid0), (x_mid1, y_mi1), 255,thickness)

        img = cv2.inpaint(img, g_mask, 7, cv2.INPAINT_NS)
        im=img
    return im
if file is None:
    st.text("Please upload an image file")
else:
    
    image = Image.open(file)
    st.image(image,use_column_width=False)
    image=cv2.cvtColor(np.uint8(image), cv2.COLOR_BGR2RGB)
    resize_image=preprocess_image(image)
    pred=model.predict_classes(resize_image)
    prediction=findout(pred)
    if prediction == 0:
       st.write("It is a COVID!")
    elif prediction == 1:
        st.write("It is a Pneumonia!")
    elif prediction == 2:
         st.write("It is a Lung Opacity!")
    else:
       st.write("It is Normal!")
    
    st.text("Probability (0: COVID, 1: Pneumonia, 2: Lung Opacity, 3: Normal")
    st.write(prediction)
