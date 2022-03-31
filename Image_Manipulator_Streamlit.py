import streamlit as st
import cv2
import numpy as np
import random

st.write("# Image Manipulator")

uploaded_file = st.file_uploader("Choose an Image", type=['png','jpg','jpeg'])

if uploaded_file is not None:
     # To read file as bytes:
     bytes_data = uploaded_file.getvalue()
     img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

     option = st.selectbox(
     'Select your Filter',
     ("Grey", "Edge Detection", "Gaussian Blur", "Contour", "Pencil Sketch"))

     st.write('You selected:', option)

     col1, col2 = st.columns(2)
     
     col1.image(img, channels="BGR")


     if option == "Grey":
          grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

          col2.image(grey_img)

     elif option == "Edge Detection":
          edge_img = cv2.Canny(img,50,50)

          col2.image(edge_img)

     elif option == "Gaussian Blur":
          Gauss = cv2.GaussianBlur(img, (9, 9), 0)

          col2.image(Gauss, channels="BGR")

     elif option == "Contour":
          def thresh_callback(val):
               threshold = val
               # Detect edges using Canny
               canny_output = cv2.Canny(src_gray, threshold, threshold * 2)
               # Find contours
               contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
               
               drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
               for i in range(len(contours)):
                    color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
                    cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)
              

               col2.image(drawing)

          src_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          thresh = 100 # initial threshold
          thresh_callback(thresh)

     elif option == "Pencil Sketch":
          grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
          invert_img = cv2.bitwise_not(grey_img)
          blur_img = cv2.GaussianBlur(invert_img, (111,111), 0)
          invblur_img = cv2.bitwise_not(blur_img)
          sketch_img = cv2.divide(grey_img, invblur_img, scale=256.0)

          col2.image(sketch_img)
