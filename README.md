# vaultlayer-examples

Canonical training scripts demonstrating workloads under [VaultLayer](https://vaultlayer.ai) protection.

All scripts implement the [job contract](https://github.com/hector25/vaultlayer/blob/main/docs/job_contract.md) — they read `VAULTLAYER_CHECKPOINT_DIR`, save in HuggingFace `checkpoint-N/` format, and call `trainer.train(resume_from_checkpoint=...)` so cross-provider failover resumes mid-run.

## Stress-test scripts (real LLM fine-tunes)

| Script | Model | VRAM | Duration | Stresses |
|---|---|---|---|---|
| `train_tinyllama_qlora.py` | TinyLlama-1.1B | ~3 GB | ~10 min on A10G | Smoke. Fits any GPU. |
| `train_llama3_2_3b_qlora.py` | Llama-3.2-3B-Instruct (gated) | ~8 GB | ~25-40 min on A10G | Mid-size weights, multi-GB checkpoint save/restore |
| `train_qwen7b_qlora.py` | Qwen2.5-7B-Instruct (ungated) | ~16 GB | ~12-20 min on A100-40 | Production-class 7B, no HF_TOKEN needed |
| `train_mistral7b_qlora.py` | Mistral-7B-v0.1 (gated) | ~15 GB | ~12-20 min on A100-40 | Alternative 7B arch. Requires HF_TOKEN. |

## Demo scripts

| Script | What it proves |
|---|---|
| `demo_resume.py` | Cross-provider resume: kills a mid-run training job on one provider, verifies the new leg resumes from the last R2 checkpoint on a different provider. |

## Legacy scripts (still supported, smaller)

| Script | Duration | What it does | GPU |
|---|---|---|---|
| `train_quick.py` | ~30 s | GPU detection, basic tensor op, one checkpoint. Smoke test. | Any |
| `train_mnist.py` | ~2 min | Real CNN on MNIST (12 MB auto-download). | A10G or larger |
| `train_long.py` | ~10 min | Long-running workload for spot interruption / failover tests. | A10G or larger |
| `tinyllama_alpaca.py` | ~45 min | Full (non-QLoRA) fine-tune — older, heavier, kept for back-compat. | A100_40 recommended |

## Usage

```bash
# Smoke test
vaultlayer run python train_tinyllama_qlora.py

# 3B stress test
vaultlayer run --gpu A10G --env HF_TOKEN=hf_... python train_llama3_2_3b_qlora.py

# 7B stress test (no HF token needed)
vaultlayer run --gpu A100_40 --failover python train_qwen7b_qlora.py

# Resume proof
VL_TOKEN=vl_live_... python demo_resume.py
```

## Job contract

All scripts in this repo follow the [VaultLayer job contract](https://github.com/hector25/vaultlayer/blob/main/docs/job_contract.md). Key points:

- **Checkpoints**: save to `$VAULTLAYER_CHECKPOINT_DIR` as HuggingFace `checkpoint-<step>/` subdirs. The broker rsyncs this dir to R2 every 60s.
- **Resume**: on a new leg (after failover), the broker pre-populates `$VAULTLAYER_CHECKPOINT_DIR` from R2 before training starts. Scripts detect the most recent `checkpoint-*` subdir and pass it as `resume_from_checkpoint=...`.
- **Heartbeat**: the broker's bootstrap wrapper POSTs log updates every 30s. Scripts don't need to do anything special — just print regularly (use `flush=True`).

## License

MIT. Free to fork and modify for your own workloads.

## Contributing

PRs welcome. New scripts must:

- Read `$VAULTLAYER_CHECKPOINT_DIR` (not a hard-coded path).
- Save via HF Trainer's `save_steps=` (or equivalent `checkpoint-<int>/` directory format).
- Use `trainer.train(resume_from_checkpoint=<latest_checkpoint_dir>)` so cross-provider hops resume cleanly.
- Include a header docstring with VRAM + duration + GPU guidance.
- Print with `flush=True` for heartbeat visibility.
