import torch
import torch.nn as nn
import torch.nn.functional as F

latent_dim = 128
num_classes = 10
image_size = 28 * 28  # 784


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim + num_classes, 512)
        self.fc2 = nn.Linear(512 + num_classes, 1024)
        self.fc3 = nn.Linear(1024 + num_classes, image_size)

    def forward(self, z, y):
        z = torch.cat([z, y], dim=1)
        h = F.relu(self.fc1(z))
        h = torch.cat([h, y], dim=1)
        h = F.relu(self.fc2(h))
        h = torch.cat([h, y], dim=1)
        return torch.sigmoid(self.fc3(h))