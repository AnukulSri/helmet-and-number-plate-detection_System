import streamlit as st
import cv2
import numpy as np
import os
from PIL import Image
import time
import imutils
from keras.models import load_model
import base64
from streamlit_option_menu import option_menu
import mysql.connector

st.set_page_config(page_title="Helmet & Number Plate Detection", layout="centered", page_icon=":blue_car:")

st.markdown("""
<style>
.css-9s5bis.edgvbvh3{
    visibility :hidden;
}
.css-1q1n0ol.egzxvld0{
    visibility : hidden;
}
</style>
""", unsafe_allow_html=True)

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('front.png') 

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Detect", "About", "Admin"],
        default_index=0,
        icons=["cart-dash-fill", "file-text", "person-fill"],
        menu_icon="cast",
    )
 
st.markdown(
    """
    <style>
   
    .footer {
        margin-top: 330px;
        padding: 10px 0;
        text-align: center;
        font-size: 16px;
        color: grey;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.sidebar.markdown(
    """ 
    <div class="footer">
        &copy; 2024 Anukul Srivastava
    </div>
    """,
    unsafe_allow_html=True
)

if selected == "Detect":
    # Load YOLO model
    net = cv2.dnn.readNet("yolov3-custom_7000.weights", "yolov3-custom.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    # Get the names of all the layers in the network
    layer_names = net.getLayerNames()

    # Get the output layer indices and handle them appropriately
    unconnected_out_layers = net.getUnconnectedOutLayers()

    # Correctly handle indices for different OpenCV versions
    if isinstance(unconnected_out_layers[0], list) or isinstance(unconnected_out_layers[0], np.ndarray):
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]

    # Load helmet detection model
    model = load_model('helmet-nonhelmet_cnn.h5')

    st.title("Bike's Helmet and Number Plate Detection System")

    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi"])
    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        temp_file_path = os.path.join("temp", uploaded_file.name)
        with open(temp_file_path, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        video = cv2.VideoCapture(temp_file_path)

        if not video.isOpened():
            st.error("Error: Could not open video file.")
        else:
            stframe = st.empty()

            # Connect to MySQL database
            conn = mysql.connector.connect(
                host="localhost",
                user="sqluser",
                password="password",
                database="python_db"
            )
            cursor = conn.cursor()

            # Create table for storing number plate images
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS Plates (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    image LONGBLOB NOT NULL
                )
            ''')
            conn.commit()

            while True:
                ret, frame = video.read()

                if not ret:
                    break

                frame = imutils.resize(frame, height=500)
                height, width = frame.shape[:2]

                blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
                net.setInput(blob)
                outs = net.forward(output_layers)

                confidences = []
                boxes = []
                classIds = []

                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.3:
                            center_x = int(detection[0] * width)
                            center_y = int(detection[1] * height)
                            w = int(detection[2] * width)
                            h = int(detection[3] * height)
                            x = int(center_x - w / 2)
                            y = int(center_y - h / 2)
                            boxes.append([x, y, w, h])
                            confidences.append(float(confidence))
                            classIds.append(class_id)

                indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

                for i in range(len(boxes)):
                    if i in indexes:
                        x, y, w, h = boxes[i]
                        if classIds[i] == 0:  # bike
                            helmet_roi = frame[max(0, y):max(0, y) + max(0, h) // 4, max(0, x):max(0, x) + max(0, w)]
                            if helmet_roi.shape[0] > 0 and helmet_roi.shape[1] > 0:
                                helmet_roi = cv2.resize(helmet_roi, (224, 224))
                                helmet_roi = np.array(helmet_roi, dtype='float32')
                                helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
                                helmet_roi = helmet_roi / 255.0
                                prediction = int(model.predict(helmet_roi)[0][0])
                                if prediction == 0:
                                    frame = cv2.putText(frame, 'Helmet', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                        (0, 255, 0), 2)
                                else:
                                    frame = cv2.putText(frame, 'No Helmet', (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                                        (0, 0, 255), 2)

                                    # Save the detected number plate image to the database if "No Helmet" is detected
                                    plate_roi = frame[y:y + h, x:x + w]

                                    # Check if plate_roi is valid and not empty
                                    if plate_roi.size > 0:
                                        _, buffer = cv2.imencode('.jpg', plate_roi)
                                        plate_image_blob = buffer.tobytes()  # Convert to bytes for BLOB insertion

                                        # Insert the binary image into the database
                                        cursor.execute("INSERT INTO Plates (image) VALUES (%s)", (plate_image_blob,))
                                        conn.commit()
                                    else:
                                        # st.warning("Warning: Detected empty ROI, skipping this frame.")
                                        pass

                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                stframe.image(frame, channels="BGR", use_column_width=True)

            video.release()
            os.remove(temp_file_path)
            conn.close()

if selected == "About":
    st.markdown("<h1 style ='text-align:center;color:Black'>About</h1>", unsafe_allow_html=True)
    st.markdown("<p style = 'color:Black; font-size: 20px;'>We are at the forefront of leveraging Machine Learning (ML) to enhance safety and compliance on roads...</p>", unsafe_allow_html=True)

if selected == "Admin":
    tabs_font_css = """
<style>
div[class*="stTextInput"] label {
  font-size: 21px;
  color: Black;
}
</style>
"""
    st.write(tabs_font_css, unsafe_allow_html=True)
    st.markdown("<h1 style ='text-align:center;color:black'>Admin Portal</h1>", unsafe_allow_html=True)
    conn = mysql.connector.connect(
                host="localhost",
                user="sqluser",
                password="password",
                database="python_db")
    cursor = conn.cursor()
    cursor.execute("Select Name from admin_table where id = 1")
    td = cursor.fetchone()
    for i in td:
      s = i
    cursor.execute("Select password from admin_table where id = 1")
    tp = cursor.fetchone()
    for j in tp:
      p = j
    conn.commit()
    conn.close()
    
    with st.form("Information Form", clear_on_submit=True):
        Name = st.text_input("Enter Name")
        Password = st.text_input("Enter Password", type="password")

        button = st.form_submit_button("Submit")
        if button: 
            if Name == s and Password == p:
                st.write(f''' <a target="_self" href="http://localhost:8503/">
                        <button>Go to </button>
                        </a>
                        ''',
                         unsafe_allow_html=True)
            else:
                st.error("Please fill the correct Details")
