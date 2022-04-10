import streamlit as st
import cv2
import numpy as np
import random
import mediapipe as mp
from PIL import Image
from io import BytesIO
import base64

st.write("# Image Manipulator")

st.write("###### * Use Wide Mode for a better view. You can find it under Settings > Tick Wide mode.")

uploaded_file = st.file_uploader("Choose an Image", type=['png','jpg','jpeg'])

@st.cache
def grey_image(image):

     grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     return grey_image
     
@st.cache
def gaussian_image(image, value1, value2):

     gauss_image = cv2.GaussianBlur(image, (value1, value2), 0)

     return gauss_image

@st.cache
def canny_image(image, value1, value2):

     canny_image = cv2.Canny(image, value1, value2)

     return canny_image

def slider():

     value1, value2 = st.slider("Adjust your paramters:", 1, 255, (51, 101), 2)

     return value1,value2

def get_image_download_link(img,filename,text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href =  f'<a href="data:file/txt;base64,{img_str}" download="{filename}">{text}</a>'
    return href

if uploaded_file is not None:

     bytes_data = uploaded_file.getvalue()
     image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

     option = st.selectbox(
     'Select your Filter',
     ("Grey", "Edge Detection", "Gaussian Blur", "Contour", "Pencil Sketch", "Inverse Edge Detection", "Face Detection", "Face Mesh", "Pose Estimation"))

     col1, col2 = st.columns(2)

     if option == "Grey":

          col1.image(image, channels="BGR")
          col2.image(grey_image(image))

     elif option == "Edge Detection":

          value1, value2 = slider()
          edge_image = canny_image(image, value1, value2)

          col1.image(image, channels="BGR")
          col2.image(edge_image)

     elif option == "Gaussian Blur":

          value1, value2 = slider()

          col1.image(image, channels="BGR")
          col2.image(gaussian_image(image, value1, value2), channels="BGR")

     elif option == "Contour":

          src_gray = grey_image(image)
     
          value1, value2 = slider()

          canny_output = canny_image(src_gray, value1, value2 * 2)
          contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

          drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

          for i in range(len(contours)):
               color = (random.randint(0,256), random.randint(0,256), random.randint(0,256))
               cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

          col1.image(image, channels="BGR")
          col2.image(drawing)
          

     elif option == "Pencil Sketch":

          value1, value2 = slider()

          grey_image = grey_image(image)
          invert_image = cv2.bitwise_not(grey_image)
          blur_image = gaussian_image(invert_image, value1, value2)
          invblur_image = cv2.bitwise_not(blur_image)
          sketch_image = cv2.divide(grey_image, invblur_image, scale=256.0)

          col1.image(image, channels="BGR")
          col2.image(sketch_image)
          
     elif option == "Inverse Edge Detection":

          value1, value2 = slider()

          invedge_image = canny_image(image, value1, value2)
          invedge_image = cv2.bitwise_not(invedge_image)

          col1.image(image, channels="BGR")
          col2.image(invedge_image)

     elif option == "Face Detection":

          mp_face_detection = mp.solutions.face_detection
          mp_drawing = mp.solutions.drawing_utils

          face_image = image.copy()

          with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence = 0.5) as face_detection:

               results = face_detection.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

               if results.detections is None:
                    st.markdown("Sorry ! I can't find any faces.")
                    st.markdown("Could you just try with another image, please ?")
                    
               else:
                    st.markdown("I detected the face(s).")
                    for detection in results.detections:

                         mp_drawing.draw_detection(face_image, detection)

          col1.image(image, channels="BGR")
          col2.image(face_image, channels="BGR")

     elif option == "Face Mesh":

          mp_drawing = mp.solutions.drawing_utils
          mp_drawing_styles = mp.solutions.drawing_styles
          mp_face_mesh = mp.solutions.face_mesh

          drawing_spec = mp_drawing.DrawingSpec(thickness = 1, circle_radius = 1)
          with mp_face_mesh.FaceMesh(static_image_mode = True, max_num_faces = 7, refine_landmarks = True, min_detection_confidence = 0.5) as face_mesh:

               results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

               face_mesh_image = image.copy()

               if results.multi_face_landmarks is None:
                    st.markdown("Sorry ! I can't find any faces.")
                    st.markdown("Could you just try with another image, please ?")

               else:
                    st.markdown("I detected the face(s).")
                    for face_landmarks in results.multi_face_landmarks:

                         mp_drawing.draw_landmarks(image = face_mesh_image, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_TESSELATION, landmark_drawing_spec = None, connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_tesselation_style())

                         mp_drawing.draw_landmarks(image = face_mesh_image, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_CONTOURS, landmark_drawing_spec = None, connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_contours_style())

                         mp_drawing.draw_landmarks(image = face_mesh_image, landmark_list = face_landmarks, connections = mp_face_mesh.FACEMESH_IRISES, landmark_drawing_spec = None, connection_drawing_spec = mp_drawing_styles.get_default_face_mesh_iris_connections_style())

          col1.image(image, channels="BGR")
          col2.image(face_mesh_image, channels="BGR")

     elif option == "Pose Estimation":

          mp_drawing = mp.solutions.drawing_utils
          mp_drawing_styles = mp.solutions.drawing_styles
          mp_pose = mp.solutions.pose

          with mp_pose.Pose(static_image_mode = True, model_complexity = 2, enable_segmentation = True, min_detection_confidence = 0.5) as pose:

               results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

               pose_image = image.copy()

               if not results.pose_landmarks:
                    st.markdown("Sorry ! I can't find any pose(s).")

               else:

                    mp_drawing.draw_landmarks(pose_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, landmark_drawing_spec = mp_drawing_styles.get_default_pose_landmarks_style())

          col1.image(image, channels= "BGR")
          col2.image(pose_image, channels = "BGR")


          #result = Image.fromarray(pose_image)
          #st.markdown(get_image_download_link(result,"Output.jpg",'Download '+"Output.jpg"), unsafe_allow_html=True)
          #from io import BytesIO
          #buf = BytesIO()
          #buf = Image.fromarray(pose_image)
          #pose_image.save(buf, format="JPEG")
          #byte_im = buf.getvalue()

          #btn = col2.download_button(label="Download Image", data=buf, file_name="imagename.png", mime="image/png")

          #img = Image.fromarray(pose_image)

          #btn = st.download_button(label="Download image", data=img, file_name="flower.png", mime="image/png")

