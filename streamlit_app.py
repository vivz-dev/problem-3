import streamlit as st
import torch
from model import Decoder
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Digit Generator", page_icon="🔢")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using trained model.")

digit = st.selectbox("Choose a digit to geneate (0-9):", options=list(range(10)), index=0)

latent_dim = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    decoder = Decoder().to(device)
    decoder.load_state_dict(torch.load("decoder.pt", map_location=device))
    decoder.eval()
    return decoder

decoder = load_model()

if st.button("Generate images"):
    st.subheader(f"Generate images of digit {digit}")
    y = torch.eye(10)[digit].unsqueeze(0).repeat(5,1).to(device)
    z = torch.randn(5, latent_dim).to(device)

    with torch.no_grad():
        outputs = decoder(z,y).cpu().numpy().reshape(-1, 28, 28)

    fig, axs = plt.subplots(1, 5, figsize=(10,2))
    for i in range(5):
        axs[i].imshow(outputs[i], cmap = "gray")
        axs[i].axis("off")
        axs[i].set_title(f"Sample {i+1}", fontsize=8, pad=4)
    
    st.pyplot(fig)
