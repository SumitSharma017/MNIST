import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model

st.title("MNIST Digit Classifier")
st.write("🎨 Draw a digit (0-9) below and press 'Predict'")

# Load model
@st.cache_resource
def load_mnist_model():
    return load_model("mnist_digit_model.keras")  # or .h5 if that's what you saved

model = load_mnist_model()

# Draw canvas
canvas_result = st_canvas(
    fill_color="white",  # stroke will paint this color
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict only when the user clicks the button
if canvas_result.image_data is not None:
    if st.button("Predict"):
        
# Convert to grayscale
        img = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('L')

# Invert colors (white background, black digit)
        img = ImageOps.invert(img)

# Resize to 28x28
        img = img.resize((28, 28))

# Convert to numpy array, normalize
        img_array = np.array(img) / 255.0

# Reshape to (1, 28, 28, 1) for model input
        img_array = img_array.reshape(1, 28, 28, 1)

        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)

        st.success(f"🧠 Prediction: **{predicted_class}**")
