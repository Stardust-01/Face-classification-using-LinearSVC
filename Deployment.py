import streamlit as st
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import io
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import joblib

# Check if a GPU is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pre-trained ResNet-50 model and move it to the device
resnet = models.resnet50(pretrained=True).to(device)
# Remove the last fully connected layer
resnet = nn.Sequential(*list(resnet.children())[:-1])
# Set the model to evaluation mode
resnet.eval()

# Function to perform CNN on the input image
def perform_cnn(image):
    # Load and preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocess(image).unsqueeze(0).to(device)
    
    # Extract features
    with torch.no_grad():
        features = resnet(image)
    
    # Remove the batch dimension and move the features to the CPU
    features = features.squeeze(0).cpu().numpy()
    return features

# Function to perform PCA on the extracted features
def perform_pca(features):
    # Log the dimensions of the input features
    st.write("Input features dimensions:", features.shape)
    
    # Check if features are empty or have unexpected dimensions
    if features.size == 0 or features.ndim != 2:
        st.error("Invalid input features for PCA")
        return None
    
    pca = PCA(n_components=128)
    pca_features = pca.fit_transform(features)
    
    # Log the dimensions of the PCA-transformed features
    st.write("PCA-transformed features dimensions:", pca_features.shape)
    
    return pca_features

# Streamlit UI
st.title("Image Classification with CNN and PCA")
uploaded_file = st.file_uploader("Upload an image(250*250)...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Perform CNN
    cnn_features = perform_cnn(image)

    # Perform PCA
    pca_features = perform_pca(cnn_features)

    # Load the Linear SVC model
    model = joblib.load('linear_svc_128_features.pkl') 

    # Get decision scores from the model
    decision_scores = model.decision_function(pca_features)

    # Get indices of top 5 scores
    top5_indices = np.argsort(decision_scores)[0][-5:][::-1]

    # Get corresponding classes
    top5_classes = model.classes_[top5_indices]

    # Display the top 5 predictions
    st.write("Top 5 Predictions:")
    for i, cls in enumerate(top5_classes, 1):
        st.write(f"Prediction {i}: {cls}")
