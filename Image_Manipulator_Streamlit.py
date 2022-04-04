import streamlit as st
import cv2
import numpy as np
import random

st.write("# Image Manipulator")

uploaded_file = st.file_uploader("Choose an Image", type=['png','jpg','jpeg'])

@st.cache
def grey_image(img):

     grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

     return grey_img
     
@st.cache
def gaussian_image(img, value1, value2):

     gauss_img = cv2.GaussianBlur(img, (value1, value2), 0)

     return gauss_img

@st.cache
def canny_image(img, value1, value2):

     canny_img = cv2.Canny(img, value1, value2)

     return canny_img

def slider():

     value1, value2 = st.slider("Adjust your paramters:", 1, 255, (51, 101), 2)

     return value1,value2

if uploaded_file is not None:

     bytes_data = uploaded_file.getvalue()
     img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

     option = st.selectbox(
     'Select your Filter',
     ("Grey", "Edge Detection", "Gaussian Blur", "Contour", "Pencil Sketch", "Inverse Edge Detection"))

     col1, col2 = st.columns(2)

     if option == "Grey":

          col1.image(img, channels="BGR")
          col2.image(grey_image(img))

     elif option == "Edge Detection":

          value1, value2 = slider()
          edge_img = canny_image(img, value1, value2)

          col1.image(img, channels="BGR")
          col2.image(edge_img)

     elif option == "Gaussian Blur":

          value1, value2 = slider()

          col1.image(img, channels="BGR")
          col2.image(gaussian_image(img, value1, value2), channels="BGR")

     elif option == "Contour":

          src_gray = grey_image(img)
     
          value1, value2 = slider()

          canny_output = canny_image(src_gray, value1, value2 * 2)
          contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

          for i in range(len(contours)):
               color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
               cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

          col1.image(img, channels="BGR")
          col2.image(drawing)
          

     elif option == "Pencil Sketch":

          value1, value2 = slider()

          grey_image = grey_image(img)
          invert_image = cv2.bitwise_not(grey_image)
          blur_image = gaussian_image(invert_image, value1, value2)
          invblur_image = cv2.bitwise_not(blur_image)
          sketch_image = cv2.divide(grey_image, invblur_image, scale=256.0)

          col1.image(img, channels="BGR")
          col2.image(sketch_image)
          
     elif option == "Inverse Edge Detection":

          value1, value2 = slider()

          invedge_image = canny_image(img, value1, value2)
          invedge_image = cv2.bitwise_not(invedge_image)

          col1.image(img, channels="BGR")
          col2.image(invedge_image)
