"""
VaultLayer MNIST Test — trains a CNN on MNIST for 2 epochs (~2 min).
Usage: vaultlayer run --gpu A10G python vaultlayer-examples/train_mnist.py
MNIST downloads automatically (12 MB). No setup needed.
"""
import os, time, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

print("=" * 60)
print("VaultLayer MNIST Training")
print("=" * 60)

EPOCHS, BATCH_SIZE, CKPT_EVERY, LR = 2, 64, 200, 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

data_dir = os.environ.get("VL_DATA_ROOT", "/tmp") + "/mnist"
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
test_ds = datasets.MNIST(data_dir, train=False, transform=transform)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=1000)
print(f"Dataset: {len(train_ds)} train, {len(test_ds)} test")

class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1, self.conv2 = nn.Conv2d(1,16,3,1), nn.Conv2d(16,32,3,1)
        self.fc1, self.fc2 = nn.Linear(32*5*5, 128), nn.Linear(128, 10)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        return self.fc2(F.relu(self.fc1(x.view(x.size(0), -1))))

model = SmallCNN().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
print(f"Model: SmallCNN ({sum(p.numel() for p in model.parameters()):,} params)")

start_step, start_epoch = 0, 0
try:
    from vaultlayer_checkpoint import restore, CHECKPOINT_DIR
    start_step = restore(CHECKPOINT_DIR, model=model, optimizer=optimizer)
    if start_step > 0:
        start_epoch = start_step // len(train_loader)
        print(f"Resumed from step {start_step}")
except ImportError: pass

print(f"\nTraining {EPOCHS} epochs (checkpoint every {CKPT_EVERY} steps)...\n")
global_step, t0 = start_step, time.time()
for epoch in range(start_epoch, EPOCHS):
    model.train(); epoch_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        loss = F.cross_entropy(model(data), target)
        loss.backward(); optimizer.step()
        global_step += 1; epoch_loss += loss.item()
        if global_step % 100 == 0: print(f"  Epoch {epoch+1}/{EPOCHS}  Step {global_step}  Loss={loss.item():.4f}")
        if global_step % CKPT_EVERY == 0:
            try:
                from vaultlayer_checkpoint import checkpoint, CHECKPOINT_DIR
                checkpoint(step=global_step, model=model, optimizer=optimizer, save_path=CHECKPOINT_DIR, loss=loss.item())
                print(f"  >> Checkpoint at step {global_step}")
            except ImportError: pass
    print(f"  Epoch {epoch+1} done  |  Avg loss: {epoch_loss/len(train_loader):.4f}")

model.eval(); correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        correct += (model(data).argmax(1) == target).sum().item()
accuracy = 100.0 * correct / len(test_ds)

print(f"\n{'='*60}")
print(f"DONE  {EPOCHS} epochs, {global_step} steps, {time.time()-t0:.0f}s  |  Accuracy: {accuracy:.1f}%")
print(f"{'='*60}")
