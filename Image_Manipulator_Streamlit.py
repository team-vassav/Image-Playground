import streamlit as st
import cv2
import numpy as np
import random
import mediapipe as mp


def slider():
    value1, value2 = st.sidebar.slider("Adjust your parameters:", 1, 255, (201, 251), 2)

    return value1, value2

def download(output_image):

    cv2.imwrite("Output.jpg", output_image)

    with open("Output.jpg", "rb") as image_contents:
        st.download_button("Download", data=image_contents, file_name="Output.jpg", mime="image/jpg")

def process(photo):
    option = st.sidebar.selectbox('Select your Filter', ('None', 'Contour', 'Edge Detection', 'Face Detection', 'Face Mesh', 'Gaussian Blur', 'Grey', 'Inverse Edge Detection', 'Invert', 'Pencil Sketch', 'Pose Estimation'))

    col1, col2 = st.columns(2)

    if option == "None":

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(photo, caption="No Filter", channels="BGR")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(photo)

    elif option == "Grey":

        grey_photo = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(grey_photo, caption="Grey")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(grey_photo)

    elif option == "Edge Detection":

        value1, value2 = slider()
        edge_image = cv2.Canny(photo, value1, value2)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(edge_image, caption="Edge Detection")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(edge_image)

    elif option == "Gaussian Blur":

        value1, value2 = slider()

        blur_photo = cv2.GaussianBlur(photo, (value1, value2), 0)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(blur_photo, channels="BGR", caption="Gaussian Blur")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(blur_photo)

    elif option == "Contour":

        src_gray = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)

        value1, value2 = slider()

        canny_output = cv2.Canny(src_gray, value1, value2 * 2)
        contours, hierarchy = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)

        for i in range(len(contours)):
            color = (random.randint(0, 256), random.randint(0, 256), random.randint(0, 256))
            cv2.drawContours(drawing, contours, i, color, 2, cv2.LINE_8, hierarchy, 0)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(drawing, caption="Contour")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(drawing)

    elif option == "Invert":

        invert_photo = cv2.bitwise_not(photo)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(invert_photo, caption="Invert")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(invert_photo)


    elif option == "Pencil Sketch":

        value1, value2 = slider()

        grey_image = cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY)
        invert_image = cv2.bitwise_not(grey_image)
        blur_image = cv2.GaussianBlur(invert_image, (value1, value2), 0)
        invblur_image = cv2.bitwise_not(blur_image)
        sketch_image = cv2.divide(grey_image, invblur_image, scale=256.0)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(sketch_image, caption="Pencil Sketch")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(sketch_image)

    elif option == "Inverse Edge Detection":

        value1, value2 = slider()

        inv_edge_image = cv2.Canny(photo, value1, value2)
        inv_edge_image = cv2.bitwise_not(inv_edge_image)

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(inv_edge_image, caption="Inverse Edge Detection")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(inv_edge_image)

    elif option == "Face Detection":

        mp_face_detection = mp.solutions.face_detection
        mp_drawing = mp.solutions.drawing_utils

        face_image = photo.copy()

        with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:

            results = face_detection.process(cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB))

            if results.detections:
                st.success("I detected the face(s).")
                for detection in results.detections:
                    mp_drawing.draw_detection(face_image, detection)

            else:
                st.text("Could you just try with another image ?")

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(face_image, channels="BGR", caption="Face Detection")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(face_image)

    elif option == "Face Mesh":

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_face_mesh = mp.solutions.face_mesh

        drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=7, refine_landmarks=True,
                                   min_detection_confidence=0.5) as face_mesh:

            results = face_mesh.process(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))

            face_mesh_image = photo.copy()

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(image=face_mesh_image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_TESSELATION,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                    mp_drawing.draw_landmarks(image=face_mesh_image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_CONTOURS,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    mp_drawing.draw_landmarks(image=face_mesh_image, landmark_list=face_landmarks,
                                              connections=mp_face_mesh.FACEMESH_IRISES,
                                              connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

                st.success("I detected the face(s).")

            else:
                st.text("Could you just try with another image ?")

        col1.image(photo, caption="Original", channels="BGR")
        col2.image(face_mesh_image, channels="BGR", caption="Face Mesh")

        answer = st.checkbox("Wanna download the output image ?")
        if answer:
            download(face_mesh_image)

    elif option == "Pose Estimation":

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing_styles = mp.solutions.drawing_styles
        mp_pose = mp.solutions.pose

        with mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                          min_detection_confidence=0.5) as pose:

            results = pose.process(cv2.cvtColor(photo, cv2.COLOR_BGR2RGB))

            pose_image = photo.copy()

            if results.pose_landmarks:

                mp_drawing.draw_landmarks(pose_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                st.success("I detected some pose(s).")

            else:
                st.text("Could you just with other image(s) ?")

            col1.image(photo, caption="Original", channels="BGR")
            col2.image(pose_image, channels="BGR", caption="Pose Estimation")

            answer = st.checkbox("Wanna download the output image ?")
            if answer:
                download(pose_image)


st.write("# Image Manipulator")

st.info("* Use Options under the left side bar.")
st.info("* Use Wide Mode for a better view. You can find it under Right Menu > Settings > Tick Wide mode.")

source = st.sidebar.radio("Select your Image input source", ("Upload", "Webcam"))

if source == "Upload":

    uploaded_file = st.sidebar.file_uploader("Choose an Image", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        process(image)
else:
    image_buffer = st.sidebar.camera_input("Take a picture")

    if image_buffer is not None:
        bytes_data = image_buffer.getvalue()
        image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        process(image)
