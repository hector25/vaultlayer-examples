"""
VaultLayer Quick Test — runs in ~30 seconds, verifies GPU + checkpoint pipeline.
Usage: vaultlayer run --gpu A10G python vaultlayer-examples/train_quick.py
"""
import os, time, torch, torch.nn as nn

print("=" * 60)
print("VaultLayer Quick Test")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
else:
    print("GPU: None (CPU)")
print(f"PyTorch: {torch.__version__}")

model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 10)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
print(f"Model: 3-layer MLP ({sum(p.numel() for p in model.parameters()):,} params)")

start_step = 0
try:
    from vaultlayer_checkpoint import restore, CHECKPOINT_DIR
    start_step = restore(CHECKPOINT_DIR, model=model, optimizer=optimizer)
    if start_step > 0: print(f"Resumed from step {start_step}")
except ImportError: pass

total_steps = 50
print(f"\nTraining steps {start_step} to {total_steps}...\n")
t0 = time.time()
for step in range(start_step, total_steps):
    x = torch.randn(32, 256, device=device)
    y = torch.randint(0, 10, (32,), device=device)
    loss = nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step + 1) % 10 == 0: print(f"  Step {step+1}/{total_steps}  loss={loss.item():.4f}")

try:
    from vaultlayer_checkpoint import checkpoint, CHECKPOINT_DIR
    checkpoint(step=total_steps, model=model, optimizer=optimizer, save_path=CHECKPOINT_DIR, loss=loss.item())
    print(f"\nCheckpoint saved at step {total_steps}")
except ImportError: pass

print(f"\n{'='*60}")
print(f"DONE  {total_steps - start_step} steps in {time.time()-t0:.1f}s  |  device={device}")
print(f"{'='*60}")
