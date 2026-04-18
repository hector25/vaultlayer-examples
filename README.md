# vaultlayer-examples

Canonical training scripts demonstrating workloads under [VaultLayer](https://vaultlayer.ai) protection.

## Scripts

| Script | Duration | What it does | GPU |
|---|---|---|---|
| `train_quick.py` | ~30 s | GPU detection, basic tensor op, one checkpoint. Smoke test. | Any |
| `train_mnist.py` | ~2 min | Real CNN on MNIST (12 MB auto-download). Checkpoints every 200 steps. | A10G or larger |
| `train_long.py` | ~10 min | Long-running workload for spot interruption / failover / back-hop tests. | A10G or larger |
| `tinyllama_alpaca.py` | ~45 min | **Real-LLM stress test.** TinyLlama-1.1B full fine-tune on Alpaca 52k. 13 × 4 GB checkpoints to R2. | A100_40 recommended |

## Usage

```bash
# One of the managed providers picks capacity automatically
vaultlayer run python train_quick.py

# Target a specific GPU
vaultlayer run --gpu A100_40GB --failover python tinyllama_alpaca.py
```

## License

MIT. Free to fork and modify for your own workloads.

## Contributing

This repo tracks the canonical `vaultlayer-examples/` folder in the main VaultLayer repo. PRs welcome; new example scripts should:

- Print with `flush=True` for heartbeat visibility
- Import `vaultlayer_checkpoint.checkpoint` / `restore` for resume-on-migration
- Run end-to-end in under 1 hour on a single GPU
- Include a brief header comment explaining what's being demonstrated
