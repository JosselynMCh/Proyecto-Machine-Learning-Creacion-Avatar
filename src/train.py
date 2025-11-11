import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from mydataset import MyDataset 
from ugatit import U_GAT_IT

# Hyperparameters
batch_size = 16
learning_rate = 0.0002
epochs = 200

# Data loading
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = MyDataset(transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Model initialization
model = U_GAT_IT().cuda()  # Move to GPU if available

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for i, (images, _) in enumerate(train_loader):
        # Move data to GPU if available
        images = images.cuda()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)  # Assuming autoencoder-like structure

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

print('Training completed!')
