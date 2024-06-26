import torch
from PIL import Image
from torchvision import transforms
import os
import torch.nn as nn
import torch.nn.functional as F

model_name = 'model_ver1.pth'

# Load the model
class SimpleDNN(nn.Module):
    def __init__(self):
        super(SimpleDNN, self).__init__()
        self.fc1 = nn.Linear(42 * 24, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 62)

    def forward(self, x):
        x = x.view(-1, 42 * 24)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

model = SimpleDNN()
model.load_state_dict(torch.load(model_name))
model.eval()

# Preprocess the images
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

image_folder = 'cropped_image'  # Adjust the folder path accordingly
output_file = 'predicted_labels.txt'

# Sort files in the folder by filename
image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.jpg')], key=lambda x: int(x.split('cropped')[1].split('.')[0]))

with open(output_file, 'w') as f:
    for filename in image_files:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            output = model(image_tensor)

        # Get the predicted label
        predicted_label = chr(torch.argmax(output).item() + 65) if torch.argmax(output).item() < 26 else chr(torch.argmax(output).item() + 97 - 26) if torch.argmax(output).item() < 52 else chr(torch.argmax(output).item() + 48 - 52)
        print(f"Predicted label for {filename}: {predicted_label}")
        f.write(f"Predicted label for {filename}: {predicted_label}\n")
