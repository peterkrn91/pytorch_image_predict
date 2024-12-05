# %%
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# %% [markdown]
# ### Initialize Model

# %%
class MNISTModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Convolutional Layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)  # Input 1x28x28, Output 32x28x28
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 32x14x14

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)  # Input 32x14x14, Output 64x14x14
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # Output 64x7x7

        # Fully Connected Layers
        self.fc1 = nn.Linear(64 * 7 * 7, 512)  # Input 64*7*7=3136, Output 512
        self.act3 = nn.ReLU()
        self.fc2 = nn.Linear(512, 10)  # Input 512, Output 10 (number of classes)

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

# %%
# training_data = datasets.MNIST(root=".", train=True, download=True, transform=ToTensor())
# training_data = datasets.FashionMNIST(root=".", train=True, download=True, transform=ToTensor()) --> uncomment to test FashionMNIST dataset (configure your own Model.)

# test_data = datasets.MNIST(root=".", train=False, download=True, transform=ToTensor())
# test_data = datasets.FashionMNIST(root=".", train=False, download=True, transform=ToTensor()) --> uncomment to test FashionMNIST dataset (configure your own Model.)

# %% [markdown]
# ### Load Trained Model from Google Colab (5 Epochs, 90% accuracy, T4 GPU)

# %%
# Load Trained Model

device = torch.device('cpu')
model = MNISTModel()

PATH = 'trained_model_acc_90.pth'
model.load_state_dict(torch.load(PATH, map_location=device))

# %% [markdown]
# ### Predict Image Locally (uncomment to test with your local files, without having to test in Flask)

# from PIL import Image
# from torchvision.transforms import ToTensor, Normalize
# import torch


# # Define the normalization transform (if used during training)
# normalize = Normalize(mean=[0.5], std=[0.5])  # Example normalization for images with pixel values in [0, 1]

# # Preprocess the test image
# test_image_path = 'test/paint-5.png'
# test_image = Image.open(test_image_path).convert('L')  # Convert to grayscale if necessary
# test_image = test_image.resize((28, 28))  # Resize to match model input size
# test_image_tensor = ToTensor()(test_image).unsqueeze(0)  # Convert to tensor and add batch dimension

# # Apply normalization if used during training
# # test_image_tensor = normalize(test_image_tensor)

# # Perform prediction
# with torch.no_grad():
#     output = model(test_image_tensor)
#     probabilities = torch.softmax(output, dim=1)
#     predicted_class = torch.argmax(probabilities, dim=1).item()

# # Display prediction result
# print(f"Predicted class: {predicted_class}")
# print(f"Output probabilities: {probabilities.squeeze().tolist()}")


# %% [markdown]
# ### Predict Image in Flask Web-App

# %%
from flask import Flask, render_template, request, jsonify
from PIL import Image
from torchvision.transforms import ToTensor, Normalize
from io import BytesIO
import torch

normalize = Normalize(mean=[0.5], std=[0.5])

def init():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')

    @app.route('/', methods=['GET'])
    def home():
        return render_template('index.html')

    @app.route('/predict', methods=['POST'])
    def predict():
        if 'image' in request.files:
            img = Image.open(BytesIO(request.files['image'].read()))
            img = img.convert('L')
            img = img.resize((28, 28))
            img = ToTensor()(img).unsqueeze(0)
            img = normalize(img)

            with torch.no_grad():
                output = model(img)
                _, predicted = torch.max(output, 1)
                prediction = predicted.item()

                return jsonify({'prediction': prediction})
        else:
            return jsonify({'error': 'No image provided.'}), 400

    return app