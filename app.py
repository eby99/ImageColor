import streamlit as st
import torch
from torchvision.transforms import ToTensor, Resize, Compose
from PIL import Image
import numpy as np
# from model import UNetAutoEncoder  # Replace with your model file import path

import torch.nn as nn

class UNetAutoEncoder(nn.Module):
    def __init__(self):
        super(UNetAutoEncoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Load the trained model
@st.cache_resource
def load_model():
    model = UNetAutoEncoder()
    model.load_state_dict(torch.load("color_autoencoder_unet.pth", map_location='cpu'))
    model.eval()  # Set to evaluation mode
    return model

model = load_model()

# Set up the image transformation
transform = Compose([Resize((150, 150)), ToTensor()])

st.title("Grayscale to Color Image Colorization")
st.write("Upload a grayscale image to colorize it using the trained model.")

# File uploader for grayscale image
uploaded_file = st.file_uploader("Choose a grayscale image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    st.image(image, caption="Uploaded Grayscale Image", use_column_width=True)
    
    # Preprocess the image
    input_image = transform(image).unsqueeze(0)  # Add batch dimension

    # Run the model on the input image
    with torch.no_grad():
        colorized_image = model(input_image).squeeze(0)  # Remove batch dimension
    
    # Post-process the output
    colorized_image = colorized_image.permute(1, 2, 0).numpy()  # Reorder for display
    colorized_image = (colorized_image * 255).astype(np.uint8)  # Scale to [0, 255]

    # Display colorized image
    st.image(colorized_image, caption="Colorized Image", use_column_width=True)
