import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

st.title("MNIST Digit Classifier")
st.write("🎨 Draw a digit (0-9) below and press 'Predict'")

@st.cache_resource
def load_mnist_model():
        return load_model("mnist_digit_model.keras")

model = load_mnist_model()

canvas_result = st_canvas(
        fill_color="white", 
        stroke_width=14,
        stroke_color="white",
        background_color="black",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
)

if canvas_result.image_data is not None:
        if st.button("Predict"):
                img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')
                img = ImageOps.invert(img)
                img = img.resize((28, 28))
                img_array = np.array(img) / 255.0
                img_array = img_array.reshape(1, 28, 28, 1)
                prediction = model.predict(img_array)
                predicted_class = np.argmax(prediction)
                st.success(f" Prediction: **{predicted_class}**")
