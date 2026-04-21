"""
VaultLayer Stress Test: Llama-3.2-3B QLoRA Fine-Tune
====================================================
Model:    meta-llama/Llama-3.2-3B-Instruct (HuggingFace, ~6 GB download)
Dataset:  tatsu-lab/alpaca (52K instructions, ~25 MB)
Method:   QLoRA 4-bit (NF4) with LoRA rank 16
VRAM:     ~8 GB (fits A10G 24 GB, A100, H100, RTX4090)
Duration: ~25-40 minutes on A10G / ~15-20 min on A100
Checkpoint: HuggingFace `checkpoint-<step>/` format every N steps

What this stress-tests (beyond TinyLlama):
  * Mid-sized model load + quantization (6 GB → 2 GB 4-bit)
  * Multi-GB model weights persist across checkpoint save/restore
  * LoRA on gate_proj / up_proj / down_proj (standard Llama-3 target modules)
  * ~3× longer wall clock — exposes heartbeat / log-forwarding bugs that
    TinyLlama's 10-min run doesn't surface.

Gated model note:
  Llama-3.2-3B-Instruct requires HuggingFace license acceptance. Set
  HF_TOKEN in the job environment:
    vaultlayer run --env HF_TOKEN=hf_... python train_llama3_2_3b_qlora.py

Usage:
    vaultlayer run --gpu A10G python examples/stress/train_llama3_2_3b_qlora.py
    vaultlayer run --gpu A100_40 --failover python examples/stress/train_llama3_2_3b_qlora.py

Environment variables:
    MAX_STEPS=200         Override training steps (default: 300)
    MODEL_ID              Override model (default: meta-llama/Llama-3.2-3B-Instruct)
    CKPT_EVERY=50         Checkpoint frequency (default: 50)
    MAX_SEQ_LEN=512       Max sequence length
    HF_TOKEN              Required for gated Llama-3 weights
"""
import os
import sys
import time

print("=" * 60)
print("VaultLayer Stress Test: Llama-3.2-3B QLoRA")
print("=" * 60)

# ── Config ───────────────────────────────────────────────────
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Llama-3.2-3B-Instruct")
DATASET_ID = "tatsu-lab/alpaca"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
LORA_R = 16
LORA_ALPHA = 32
LR = 2e-4
BATCH_SIZE = 2
GRADIENT_ACCUM = 4  # effective batch = 8

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    from datasets import load_dataset
except ImportError as e:
    print(f"\nMissing dependency: {e}")
    print("Install: pip install torch transformers peft trl datasets bitsandbytes accelerate")
    sys.exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB)")
else:
    print("WARNING: No GPU detected. Training will be extremely slow on CPU.")
print(f"PyTorch: {torch.__version__}")
print(f"Config: max_steps={MAX_STEPS}, ckpt_every={CKPT_EVERY}, seq_len={MAX_SEQ_LEN}")

# ── Checkpoint directory (from job contract §2) ──────────────
# Broker exports both VAULTLAYER_CHECKPOINT_DIR (canonical) and CHECKPOINT_DIR
# (legacy). Reads either and falls back to a scratch path if neither is set.
CKPT_DIR = os.environ.get(
    "VAULTLAYER_CHECKPOINT_DIR",
    os.environ.get(
        "CHECKPOINT_DIR",
        os.path.join(os.environ.get("VL_DATA_ROOT", "/tmp"), "vl-checkpoints", "llama3_2_3b"),
    ),
)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Checkpoints: {CKPT_DIR}")

# ── Resume detection (job contract §3 — HF checkpoint-N/ format) ──
resume_from = None
if os.path.isdir(CKPT_DIR):
    existing = sorted(
        [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    if existing:
        resume_from = os.path.join(CKPT_DIR, existing[-1])
        print(f"Resuming from: {resume_from}")

# ── Load model (4-bit quantized) ─────────────────────────────
print(f"\nLoading {MODEL_ID} (4-bit QLoRA)...")
t0 = time.time()

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    token=os.environ.get("HF_TOKEN"),
)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)
print(f"Model loaded in {time.time() - t0:.0f}s")

# ── LoRA config (standard Llama-3 target modules) ────────────
lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, lora_config)
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())
print(f"LoRA: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

# ── Dataset ──────────────────────────────────────────────────
print(f"\nLoading dataset: {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Dataset: {len(dataset)} examples")


def format_alpaca(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


# ── Training ─────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=CKPT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUM,
    learning_rate=LR,
    bf16=torch.cuda.is_available(),
    logging_steps=25,
    save_steps=CKPT_EVERY,
    save_total_limit=3,
    warmup_steps=20,
    lr_scheduler_type="cosine",
    report_to="none",
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=format_alpaca,
    max_seq_length=MAX_SEQ_LEN,
    packing=True,
)

print(f"\nTraining for {MAX_STEPS} steps (checkpoint every {CKPT_EVERY})...\n")
t_train = time.time()
trainer.train(resume_from_checkpoint=resume_from)
elapsed = time.time() - t_train

trainer.save_model(os.path.join(CKPT_DIR, "final"))
print(f"\nFinal model saved to {CKPT_DIR}/final")

print()
print("=" * 60)
print(f"DONE  {MAX_STEPS} steps in {elapsed/60:.1f} min")
print(f"Model: {MODEL_ID}")
print(f"LoRA params: {trainable:,} ({100*trainable/total:.2f}%)")
print(f"Checkpoint dir: {CKPT_DIR}")
if torch.cuda.is_available():
    print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
print("=" * 60)
