import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="Traffic Sign Classifier",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🚦 Traffic Sign Classifier</h1>', unsafe_allow_html=True)
st.markdown("### Upload a traffic sign image and let AI identify it!")
st.markdown("---")

# Class names - matches the Colab training
CLASS_NAMES = [
    '20 km/h', '30 km/h', '50 km/h', '60 km/h', '70 km/h', '80 km/h',
    '80 km/h end', '100 km/h', '120 km/h', 'No overtaking',
    'No overtaking for tracks', 'Crossroad with secondary way',
    'Main road', 'Give way', 'Stop', 'Road up', 'Road up for track',
    'Brock', 'Other dangerous', 'Turn left', 'Turn right',
    'Winding road', 'Hollow road', 'Slippery road', 'Narrowing road',
    'Roadwork', 'Traffic light', 'Pedestrian', 'Children', 'Bike',
    'Snow', 'Deer', 'End of the limits', 'Only right', 'Only left',
    'Only straight', 'Only straight and right', 'Only straight and left',
    'Take right', 'Take left', 'Circle crossroad',
    'End of overtaking limit', 'End of overtaking limit for track'
]

# Sidebar
with st.sidebar:
    st.header(" About")
    st.write("""
    This AI model can identify **43 different types** of German traffic signs.
    
    **How to use:**
    1. Upload an image
    2. Click "Classify"
    3. See the prediction!
    """)
    
    st.markdown("---")
    
    st.header(" Model Info")
    st.info("""
    - **Classes:** 43
    - **Input Size:** 32x32 pixels
    - **Architecture:** CNN
    - **Trained on:** GTSRB Dataset
    """)
    
    st.markdown("---")
    st.write("Made using Streamlit & TensorFlow")

# Load model with proper error handling
@st.cache_resource
def load_model():
    try:
        with st.spinner(" Loading AI model..."):
            # Try loading with compile=False first
            model = tf.keras.models.load_model(
                'traffic_sign_classifier_final.h5',
                compile=False
            )
            # Recompile with current TensorFlow version
            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
        st.sidebar.success(" Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(" Model file not found! Please ensure 'traffic_sign_classifier_final.h5' is in the same folder as app.py")
        return None
    except Exception as e:
        st.error(f" Error loading model: {e}")
        st.info(" Try retraining the model with the current TensorFlow version")
        return None

model = load_model()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader(" Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a traffic sign image...",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a traffic sign"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Image info
        st.caption(f"Image size: {image.size[0]}x{image.size[1]} pixels")

with col2:
    st.subheader(" Prediction Results")
    
    if uploaded_file is None:
        st.info(" Please upload an image to get started!")
    else:
        # Classify button
        if st.button(" Classify Traffic Sign", type="primary", use_container_width=True):
            if model is not None:
                with st.spinner(" Analyzing image..."):
                    try:
                        # Preprocess image (same as training)
                        img = np.array(image)
                        img = cv2.resize(img, (32, 32))
                        img = img.astype('float32') / 255.0
                        img = np.expand_dims(img, axis=0)
                        
                        # Make prediction
                        predictions = model.predict(img, verbose=0)
                        predicted_class = np.argmax(predictions[0])
                        confidence = predictions[0][predicted_class] * 100
                        
                        # Display main prediction
                        st.success(" Classification Complete!")
                        
                        # Big result card
                        st.markdown(f"""
                        <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                    padding: 30px; border-radius: 15px; text-align: center;
                                    color: white; margin: 20px 0;'>
                            <h2 style='margin: 0; font-size: 2.5rem;'>🚦</h2>
                            <h3 style='margin: 10px 0;'>{CLASS_NAMES[predicted_class]}</h3>
                            <p style='margin: 5px 0; font-size: 1.5rem; font-weight: bold;'>{confidence:.1f}%</p>
                            <p style='margin: 0; opacity: 0.9;'>Confidence</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence bar
                        st.progress(confidence / 100)
                        
                        # Top 3 predictions
                        st.markdown("---")
                        st.markdown("** Top 3 Predictions:**")
                        
                        top_3_idx = np.argsort(predictions[0])[-3:][::-1]
                        
                        for i, idx in enumerate(top_3_idx):
                            conf = predictions[0][idx] * 100
                            emoji = ["1", "2", "3"][i]
                            
                            col_a, col_b = st.columns([3, 1])
                            with col_a:
                                st.write(f"{emoji} **{CLASS_NAMES[idx]}**")
                            with col_b:
                                st.write(f"{conf:.1f}%")
                            
                            # Progress bar for each
                            st.progress(conf / 100)
                        
                        # Additional info
                        st.markdown("---")
                        if confidence > 90:
                            st.success(" High confidence prediction!")
                        elif confidence > 70:
                            st.warning(" Moderate confidence. Consider better lighting or angle.")
                        else:
                            st.error(" Low confidence. Try a clearer image.")
                            
                    except Exception as e:
                        st.error(f" Error during classification: {e}")
            else:
                st.error(" Model not loaded. Cannot classify image.")

# Footer with tips
st.markdown("---")
st.markdown("###  Tips for Best Results:")

tip_col1, tip_col2, tip_col3 = st.columns(3)

with tip_col1:
    st.markdown("""
    **Good Lighting**
    - Use natural daylight
    - Avoid shadows
    - Clear visibility
    """)

with tip_col2:
    st.markdown("""
    **Proper Angle**
    - Front-facing view
    - Not too tilted
    - Sign fills frame
    """)

with tip_col3:
    st.markdown("""
    **Image Quality**
    - Clear and sharp
    - No blur
    - Decent resolution
    """)

# Sample images section
with st.expander("Sample Traffic Signs Reference"):
    st.write("Here are examples of signs this model can recognize:")
    
    ref_col1, ref_col2, ref_col3 = st.columns(3)
    
    with ref_col1:
        st.write("**Speed Limits:**")
        st.write("- 20, 30, 50, 60, 70, 80, 100, 120 km/h")
        
    with ref_col2:
        st.write("**Warning Signs:**")
        st.write("- Pedestrians, Children, Roadwork")
        st.write("- Slippery road, Wild animals")
        
    with ref_col3:
        st.write("**Regulatory Signs:**")
        st.write("- Stop, Give way, No entry")
        st.write("- Turn left/right, Keep right/left")

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Built with Streamlit  | Powered by TensorFlow </p>",
    unsafe_allow_html=True
)