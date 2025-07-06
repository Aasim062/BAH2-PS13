import torch
from torch.utils.data import DataLoader, Dataset
from model import PINNNet
from loss_functions import lunar_lambert_loss
import cv2
import os

# Dummy dataset (real training requires multiple image-DEM pairs)
class PINNDataset(Dataset):
    def __init__(self, image_dir, dem_dir):
        self.image_paths = sorted(os.listdir(image_dir))
        self.dem_paths = sorted(os.listdir(dem_dir))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(f'./pinn/synthetic_data/images/{self.image_paths[idx]}', 0) / 255.0
        dem = cv2.imread(f'./pinn/synthetic_data/dems/{self.dem_paths[idx]}', 0) / 255.0
        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        sun_angle_tensor = torch.full_like(img_tensor, 0.7)  # dummy fixed value
        input_tensor = torch.cat([img_tensor, sun_angle_tensor], dim=0)
        return input_tensor, torch.tensor(dem, dtype=torch.float32).unsqueeze(0)

# Training loop
model = PINNNet().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
dataset = PINNDataset('./pinn/synthetic_data/images', './pinn/synthetic_data/dems')
loader = DataLoader(dataset, batch_size=2)

for epoch in range(5):
    total_loss = 0
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        pred = model(x)
        loss = lunar_lambert_loss(pred, x[:, 0:1], torch.ones_like(pred), torch.tensor(0.7))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}: {total_loss:.4f}")
