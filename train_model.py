import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# --------------------
# CONFIG
# --------------------
DATA_DIR = "data/train"
MODEL_PATH = "models/garbage_model.pth"
BATCH_SIZE = 8
EPOCHS = 25
IMG_SIZE = 224
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --------------------
# TRANSFORMS
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# DATASET
# --------------------
dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

class_names = dataset.classes
print("Classes:", class_names)

# --------------------
# MODEL
# --------------------
model = models.mobilenet_v3_small(weights="DEFAULT")
model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
model = model.to(device)

# --------------------
# TRAINING SETUP
# --------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# --------------------
# TRAIN LOOP
# --------------------
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# --------------------
# SAVE MODEL
# --------------------
os.makedirs("models", exist_ok=True)

torch.save({
    "model_state": model.state_dict(),
    "classes": class_names
}, MODEL_PATH)

print("✅ Training complete")
print("✅ Model saved to:", MODEL_PATH)
