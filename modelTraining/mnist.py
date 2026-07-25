import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import wandb
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# 1. Initialize W&B and define hyperparameters
wandb.init(
    project="llm_from_scratch",
    config={
        "learning_rate": 0.001,
        "epochs": 5,
        "batch_size": 64,
        "architecture": "CNN"
    }
)
config = wandb.config

# 2. Prepare Data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True, transform=transform),
    batch_size=config.batch_size, shuffle=True
)

# 3. Define Simple CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv = nn.Conv2d(1, 10, kernel_size=5)
        self.fc = nn.Linear(1440, 10)

    def forward(self, x):
        x = torch.relu(torch.max_pool2d(self.conv(x), 2))
        x = x.view(-1, 1440)
        return torch.log_softmax(self.fc(x), dim=1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
criterion = nn.NLLLoss()

# 4. Training Loop
model.train()
for epoch in range(1, config.epochs + 1):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # Log metrics dynamically
        if batch_idx % 100 == 0:
            pred = output.argmax(dim=1, keepdim=True)
            correct = pred.eq(target.view_as(pred)).sum().item()
            accuracy = correct / len(data)
            
            wandb.log({"loss": loss.item(), "accuracy": accuracy})
            print(f"Epoch {epoch} | Batch {batch_idx} | Loss: {loss.item():.4f}")

# 5. Mark the run as finished
wandb.finish()
