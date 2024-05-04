import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os

# 設定圖片轉換
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor()
])

# 讀取和處理圖片數據
def load_data(path):
    images = []
    labels = []
    label_mapping = {chr(i): i - 65 for i in range(65, 91)}
    label_mapping.update({chr(i): i - 97 + 26 for i in range(97, 123)})
    label_mapping.update({chr(i): i - 48 + 52 for i in range(48, 58)})

    for filename in os.listdir(path):
        if filename.endswith('.bmp'):
            label = label_mapping[filename[0]]
            with Image.open(os.path.join(path, filename)) as img:
                tensor = transform(img)
                images.append(tensor)
                labels.append(label)
    return torch.stack(images), torch.tensor(labels)

# 定義簡單的DNN模型
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

# 加載數據
path = 'letters_numbers'  # 更改為您的圖片文件夾路徑
images, labels = load_data(path)

# 創建模型、損失函數和優化器
model = SimpleDNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 訓練模型
epochs = 50
batch_size = 4
n_samples = len(images)
n_batches = (n_samples + batch_size - 1) // batch_size

for epoch in range(epochs):
    for i in range(n_batches):
        start = i * batch_size
        end = start + batch_size
        inputs = images[start:end]
        targets = labels[start:end]
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item()}')

# 儲存模型
torch.save(model.state_dict(), 'model.pth')
print("模型已保存至 'model.pth'")
