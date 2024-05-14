import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np


data_cat=['apple','banana','beetroot','bell pepper','cabbage','capsicum',
          'carrot','cauliflower','chilli pepper','corn','cucumber',
          'eggplant','garlic','ginger','grapes','jalepeno','kiwi','lemon',
          'lettuce','mango','onion','orange','paprika','pear','peas',
          'pineapple','pomegranate','potato','raddish','soy beans',
          'spinach','sweetcorn','sweetpotato','tomato','turnip',
          'watermelon']
st.header("Image Classifination Model")
img_width=180
img_height=180

model=load_model(r"C:\Users\gokul\Desktop\DeepLearning Projects\image_clssify.keras")
#image=r"C:\Users\gokul\Desktop\DeepLearning Projects\apple2.jpg"
image=st.text_input("Enter the Image","apple3.jpg")
image_load=tf.keras.utils.load_img(image,target_size=(img_height,img_width))
img_arr=tf.keras.utils.img_to_array(image_load)
img_bat=tf.expand_dims(img_arr,0)
prediction=model.predict(img_bat)
score=tf.nn.softmax(prediction)
st.image(image, caption="Uploaded Image", use_column_width=True)
st.write("friut in image is {} with accuracy of {:0.2f}".format(data_cat[np.argmax(score)],np.max(score)*100))

