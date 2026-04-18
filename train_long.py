"""
VaultLayer Long-Running Test — ~10 minutes with regular checkpoints.
Use this to test: spot interruption, failover, back-hop, credit exhaustion.
Usage: vaultlayer run --gpu A10G --failover python vaultlayer-examples/train_long.py
"""
import os, time, torch, torch.nn as nn

print("=" * 60)
print("VaultLayer Long-Running Test (10 min)")
print("=" * 60)

TOTAL_STEPS, CKPT_EVERY, PRINT_EVERY = 10000, 500, 100
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda": print(f"GPU: {torch.cuda.get_device_name(0)}")

model = nn.Sequential(nn.Linear(512,1024), nn.ReLU(), nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024,512), nn.ReLU(), nn.Linear(512,10)).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
print(f"Model: 4-layer MLP ({sum(p.numel() for p in model.parameters()):,} params)")

start_step = 0
try:
    from vaultlayer_checkpoint import restore, CHECKPOINT_DIR
    start_step = restore(CHECKPOINT_DIR, model=model, optimizer=optimizer)
    if start_step > 0: print(f"Resumed from step {start_step}")
except ImportError: pass

print(f"\nTraining {start_step} to {TOTAL_STEPS} (checkpoint every {CKPT_EVERY} steps)\n")
t0 = time.time()
for step in range(start_step, TOTAL_STEPS):
    x = torch.randn(64, 512, device=device); y = torch.randint(0, 10, (64,), device=device)
    loss = nn.functional.cross_entropy(model(x), y)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    if (step+1) % PRINT_EVERY == 0:
        elapsed = time.time() - t0; sps = (step+1-start_step)/max(elapsed,0.1)
        print(f"  Step {step+1}/{TOTAL_STEPS}  loss={loss.item():.4f}  {sps:.0f} steps/s  ETA {(TOTAL_STEPS-step-1)/max(sps,0.1)/60:.1f}min")
    if (step+1) % CKPT_EVERY == 0:
        try:
            from vaultlayer_checkpoint import checkpoint, CHECKPOINT_DIR
            checkpoint(step=step+1, model=model, optimizer=optimizer, save_path=CHECKPOINT_DIR, loss=loss.item())
            print(f"  >> Checkpoint at step {step+1}")
        except ImportError: pass
    time.sleep(0.05)

print(f"\n{'='*60}")
print(f"DONE  {TOTAL_STEPS-start_step} steps in {(time.time()-t0)/60:.1f} min  |  device={device}")
print(f"{'='*60}")
