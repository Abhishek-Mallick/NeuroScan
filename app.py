import streamlit as st
from werkzeug.utils import secure_filename
import os
import pandas as pd
import numpy as np
import random
import shutil
from shutil import copyfile
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras import Model
from sklearn.utils import shuffle
from tqdm import tqdm
from streamlit_lottie import st_lottie
import requests
import plotly.graph_objects as go
import urllib.request

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png'}

def train_val_generators(TRAINING_DIR, VALIDATION_DIR, TEST_DIR):

    train_datagen = ImageDataGenerator(rescale=1./127.5, rotation_range=30, width_shift_range=0.2,height_shift_range=0.2, shear_range=0.2, zoom_range=0.2,horizontal_flip=True, fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(directory=TRAINING_DIR, batch_size=32,class_mode='binary', target_size=(150, 150))

    valid_or_test_datagen = ImageDataGenerator(rescale=1./127.5)

    validation_generator = valid_or_test_datagen.flow_from_directory(directory=VALIDATION_DIR, batch_size=32,class_mode='binary', target_size=(150, 150))

    test_generator = valid_or_test_datagen.flow_from_directory(directory=TEST_DIR, batch_size=32,class_mode='binary', target_size=(150, 150))
    return train_generator, validation_generator, test_generator


base_dir = 'MODELLING/'
training_dir = os.path.join(base_dir, 'training')
validation_dir = os.path.join(base_dir, 'validation')
testing_dir = os.path.join(base_dir, 'testing')

train_generator, validation_generator, test_generator = train_val_generators(training_dir, validation_dir, testing_dir)


# Load the trained model
model = tf.keras.models.load_model('Desktop/brain_tumor_V10/brain_tumor.h5')

def prediction(YOUR_IMAGE_PATH):
    img = image.load_img(YOUR_IMAGE_PATH, target_size=(150, 150))
    x = image.img_to_array(img)
    x /= 127.5
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    score = tf.nn.sigmoid(classes[0])

    class_name = train_generator.class_indices
    class_name_inverted = {y: x for x, y in class_name.items()}

    if classes[0] > 0.5:
        return class_name_inverted[1], 100 * np.max(score)
    else:
        return class_name_inverted[0], 100 * np.max(score)

lottie_gif = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_33asonmr.json")

def main():
    st.set_page_config(page_title="Neuroscan",page_icon="🧠" ,layout="wide")
    st.title("🧠 Neuroscan")
    st.subheader('Brain Tumor Detection Using Machine Learning')
    st.markdown("<hr>", unsafe_allow_html=True)
    
    col1,col2 = st.columns(2)
    with col1:
        st.subheader('Project Description')
        st.write('_Description:_ The objective of this project is to develop a machine learning model that can accurately detect brain tumors in medical imaging data, such as MRI scans. By leveraging advanced image processing techniques and machine learning algorithms, the model aims to assist healthcare professionals in the early and accurate diagnosis of brain tumors, which can lead to timely treatment interventions and improved patient outcomes.')

        with st.expander("Features"):
            st.write('''**_Dataset Collection and Preprocessing:_** The project will involve the collection of a diverse dataset of brain MRI scans, including both tumor-positive and tumor-negative cases. The dataset will be carefully curated and preprocessed to ensure data quality, consistency, and appropriate anonymization.
        
**_Image Segmentation:_** The project will utilize image segmentation techniques to isolate the regions of interest within the brain MRI scans. This step will help to distinguish the tumor regions from the healthy brain tissue and background, facilitating accurate tumor detection.

**_Feature Extraction:_** Relevant features will be extracted from the segmented brain tumor images. These features may include shape, texture, intensity, and other characteristics that can capture the unique properties of tumor regions.

**_Machine Learning Model Development:_** Various machine learning algorithms, such as convolutional neural networks (CNNs) or support vector machines (SVMs), will be explored and evaluated for their effectiveness in detecting brain tumors. The model will be trained using the labeled dataset, learning to classify brain scans as tumor-positive or tumor-negative based on the extracted features.

**_Model Evaluation and Validation:_** The developed model will be evaluated using appropriate evaluation metrics, such as accuracy, sensitivity, specificity, and area under the curve (AUC). Cross-validation techniques will be employed to ensure robustness and generalizability of the model's performance.

**_User Interface Development:_** A user-friendly interface will be developed to allow healthcare professionals to interact with the model effectively. The interface will enable them to upload brain MRI scans and receive the model's predictions regarding the presence of brain tumors.
        ''')
        with st.expander("Benefit"):
            st.write('''**_Early and Accurate Diagnosis:_** The developed model can aid healthcare professionals in detecting brain tumors at an early stage, leading to timely intervention and improved patient outcomes.
            
**_Time and Cost Efficiency:_** Automated brain tumor detection can reduce the time and effort required for manual analysis of MRI scans, enabling healthcare professionals to focus on treatment planning and patient care.

**_Improved Detection Accuracy:_** By leveraging advanced machine learning algorithms, the model can potentially achieve higher accuracy rates in brain tumor detection compared to traditional manual interpretation of scans.

**_Assistive Tool for Healthcare Professionals:_** The model serves as an assistive tool for healthcare professionals, providing additional support and insights in the diagnosis process.

**_Research Advancement:_** The project contributes to the field of medical imaging and machine learning research by exploring novel techniques for brain tumor detection and advancing the understanding of brain tumor analysis using artificial intelligence.''')
        st.write('Overall, this project aims to develop a machine learning-based solution for brain tumor detection, with the potential to assist healthcare professionals in accurate and timely diagnosis. By leveraging state-of-the-art techniques in image processing and machine learning, the project has the potential to significantly impact patient care and improve outcomes in the field of neuro-oncology.')

            
    with col2:
        st_lottie(lottie_gif)

       
    col_left,col_right=st.columns(2)
    with col_left:
        st.subheader('Prediction')
        st.write("Upload an image to predict if it contains a brain tumor.")

        uploaded_file = st.file_uploader("Choose an image file", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            if allowed_file(uploaded_file.name):
                filename = secure_filename(uploaded_file.name)
                upload_dir = 'static/uploads'
                os.makedirs(upload_dir, exist_ok=True) 
                file_path = os.path.join(upload_dir, filename)
                with open(file_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                cols1,cols2,cols3= st.columns(3)
                with cols2:
                    st.image(file_path, caption='Uploaded Image', use_column_width=True,width=200)

                prediction_result, confidence = prediction(file_path)

                if prediction_result == '0':
                    prediction_message = f"The uploaded image '{filename}' most likely does not show any brain tumor."
                else:
                    prediction_message = f"The uploaded image '{filename}' most likely shows a brain tumor."

                st.write(prediction_message)
                st.write(f"Confidence: {confidence:.2f}%")

                confidence_df = pd.DataFrame({'Class': ['Brain Tumor', 'No Brain Tumor'], 'Confidence': [confidence, 100-confidence]})
                st.bar_chart(confidence_df.set_index('Class'))

            else:
                st.write("Invalid file format. Please upload a valid image file.")

    with col_right:
        st.subheader('About Brain Tumor')
        st.write('''A brain tumor is an abnormal growth of cells in the brain. It can be either cancerous (malignant) or non-cancerous (benign). Brain tumors can arise from the brain itself (primary tumors) or spread to the brain from other parts of the body (secondary tumors or metastatic tumors).''')
        st.write('''The treatment of brain tumors depends on several factors, including the type, location, size, and grade of the tumor, as well as the overall health of the patient. The main treatment options for brain tumors include:''')
        with st.expander('Treatment'):
            st.write('''**_Surgery:_** Surgical removal of the tumor is often the first-line treatment when possible. The surgeon aims to remove as much of the tumor as safely possible while preserving brain function.

**_Radiation Therapy:_** Radiation therapy uses high-energy beams to kill or shrink tumor cells. It may be used before or after surgery or as the primary treatment for inoperable tumors.

**_Chemotherapy:_** Chemotherapy involves the use of drugs to kill cancer cells. It can be administered orally or intravenously and may be used alone or in combination with surgery and radiation therapy.

**_Targeted Therapy:_** Targeted therapy uses drugs that specifically target certain molecules or pathways involved in the growth and survival of cancer cells. These therapies can be effective in treating certain types of brain tumors.

**_Immunotherapy:_** Immunotherapy works by stimulating the body's immune system to recognize and attack cancer cells. It is a rapidly evolving field, and some immunotherapies have shown promising results in treating certain types of brain tumors.

**_Supportive Care:_** In addition to the specific treatments, supportive care is important to manage symptoms, provide pain relief, and improve the quality of life of patients with brain tumors. This may include medications, physical therapy, occupational therapy, and psychological support.''')
        st.write('''The treatment approach is individualized based on the specific characteristics of the tumor and the patient. It often involves a multidisciplinary team of healthcare professionals, including neurosurgeons, neuro-oncologists, radiation oncologists, and other specialists.

It's important to note that the treatment outcomes and prognosis for brain tumors vary widely depending on the type and stage of the tumor, as well as other individual factors. Some tumors can be successfully treated and cured, while others may be more challenging to manage. Regular follow-up and monitoring are crucial to detect any recurrence or changes in the tumor.''')
    
    st.markdown("<hr>", unsafe_allow_html=True)
    col3,col4= st.columns(2)
    with col3:
        st.subheader('👤 Profile')
        st.write('I possess a deep love for coding and a strong desire to continuously improve my skills. With a demonstrated proficiency in multiple programming languages and a natural curiosity for problem-solving, I am eager to contribute my technical expertise to any organization')
        st.markdown("<p style= font-size:16px;>Deepraj Bera</p><p style= font-size:12px;>Full Stack | ML</p><a href='https://github.com/deepraj21'><img src='https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white' alt='GitHub'></a>", unsafe_allow_html=True)
    with col4:
        st.subheader('✉️ Find Me')
        st.markdown("<a href='mailto: deepraj21.bera@gmail.com' style='text-decoration:none;'>deepraj21.bera@gmail.com</a>",unsafe_allow_html=True)
        st.markdown("<a href='mailto: 21051302@kiit.ac.in'>21051302@kiit.ac.in</a>",unsafe_allow_html=True)


    st.markdown("<center><p>© 2023 Neuroscan</p><center>",unsafe_allow_html=True)

if __name__ == '__main__':
    main()

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


