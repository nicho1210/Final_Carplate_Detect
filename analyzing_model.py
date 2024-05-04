import torch
import torch.nn as nn
import torch.nn.functional as F
import csv

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
model.load_state_dict(torch.load('model.pth'))

# Extract weights and biases from each layer
weights_biases = {}
for name, param in model.named_parameters():
    if 'weight' in name:
        weights_biases[name] = param.detach().numpy()
    elif 'bias' in name:
        weights_biases[name] = param.detach().numpy()

# Save to CSV
csv_file = 'weights_biases.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Layer', 'Weights', 'Biases'])
    for layer, values in weights_biases.items():
        writer.writerow([layer, values.tolist(), None if 'weight' in layer else values.tolist()])
