import streamlit as st 
import cv2 
import numpy as np 
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.set_page_config(page_title="Face Mask Detection", page_icon=":guardsman:", layout="centered")

model = load_model('face_mask_detector.h5')

def predict_mask(img):
    img = img.resize((128,128))
    img_array = np.array(img)
    img_array = img_array/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)
    input_pred_label = np.argmax(prediction)
    if input_pred_label ==1:
        return "Mask Detected", "green"
    else:
        return "No Mask Detected", "red"

def capture_image_from_webcam(model):
    st.title("Click a Picture to Detect Mask or No Mask")
    
    # Open the webcam
    cap = cv2.VideoCapture(0)
    
    # Show a button to capture an image
    if st.button("Capture Image"):
        ret, frame = cap.read()
        if ret:
            # Convert the captured frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Show the captured image to the user (smaller size)
            st.image(frame_rgb, channels="RGB", width=300)
            
            # Convert to PIL Image for prediction
            img = Image.fromarray(frame_rgb)

            # Predict using the model
            label, color = predict_mask(img)
            
            # Display prediction with colored text
            st.markdown(f"<h3 style='color:{color};'>{label}</h3>", unsafe_allow_html=True)
        
        # Release the webcam after capturing the image
        cap.release()
        cv2.destroyAllWindows()



st.title("😷Face Mask Detection App")
option = st.radio("Choose an Option",("Upload Image","Use Webcam"))
if option == "Upload Image":
    st.subheader("Upload an Image to detect Mask or No Mask")
    uploaded_file = st.file_uploader("Choose an image.....",type=["jpg","jpeg","png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=False, width=300)
        label,color = predict_mask(img)
        if label == "Mask Detected":
            st.success(label, icon="✅")
        else:
            st.error(label, icon="❌")
        
elif option == "Use Webcam":
    capture_image_from_webcam(model=model)
        