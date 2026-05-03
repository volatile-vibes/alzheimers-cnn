import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

# Define CNN (same as training)
class AlzheimerCNN(nn.Module):
    def __init__(self):
        super(AlzheimerCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 4)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = AlzheimerCNN().to(device)
model.load_state_dict(torch.load("alz_model.pth", map_location=device))
model.eval()

# Define transforms
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Class names (adjust to your dataset order)
class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]

# Load and preprocess single image
img_path = "no_1.jpg"   # replace with your test image path
image = Image.open(img_path)
image = transform(image).unsqueeze(0).to(device)

# Predict
with torch.no_grad():
    output = model(image)
    _, predicted = torch.max(output.data, 1)

print("Predicted Class:", class_names[predicted.item()])
