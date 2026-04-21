"""
VaultLayer Stress Test: Mistral-7B QLoRA Fine-Tune
==================================================
Model:    mistralai/Mistral-7B-v0.1 (HuggingFace, GATED, ~14 GB download)
Dataset:  tatsu-lab/alpaca (52K instructions)
Method:   QLoRA 4-bit (NF4) with LoRA rank 16
VRAM:     ~15 GB (fits A100-40, A100-80, H100, RTX4090)
Duration: ~12-20 minutes on A100-40
Checkpoint: HuggingFace `checkpoint-<step>/` format every N steps

Gated model note:
  Mistral-7B requires HuggingFace license acceptance. Set HF_TOKEN in the job
  environment (accept license at https://huggingface.co/mistralai/Mistral-7B-v0.1):
    vaultlayer run --env HF_TOKEN=hf_... python train_mistral7b_qlora.py

  If you can't acquire a token, use `train_qwen7b_qlora.py` — same size class,
  comparable architecture, no gating. Matrix row 4b proves the same
  broker/failover paths exercise the same code.

Known-good versions (from docker/training-base.Dockerfile):
  transformers>=4.41.2  (older tokenizer versions crash on Mistral's tokenizer.json)
  tokenizers>=0.19
  accelerate>=0.30
  peft>=0.11
  trl>=0.8

Usage:
    vaultlayer run --gpu A100_40 --env HF_TOKEN=hf_... python examples/stress/train_mistral7b_qlora.py
"""
import os
import sys
import time

print("=" * 60)
print("VaultLayer Stress Test: Mistral-7B QLoRA")
print("=" * 60)

MODEL_ID = os.environ.get("MODEL_ID", "mistralai/Mistral-7B-v0.1")
DATASET_ID = "tatsu-lab/alpaca"
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "50"))
MAX_SEQ_LEN = int(os.environ.get("MAX_SEQ_LEN", "512"))
LORA_R = 16
LORA_ALPHA = 32
LR = 2e-4
BATCH_SIZE = 1
GRADIENT_ACCUM = 8

HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN and MODEL_ID.startswith("mistralai/"):
    print("\nWARNING: MODEL_ID is a Mistral gated model and HF_TOKEN is not set.")
    print("The model download will likely fail with HTTP 401.")
    print("Either set HF_TOKEN or use train_qwen7b_qlora.py (ungated, same size class).")

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
print(f"PyTorch: {torch.__version__}")
print(f"Config: max_steps={MAX_STEPS}, ckpt_every={CKPT_EVERY}, seq_len={MAX_SEQ_LEN}")

CKPT_DIR = os.environ.get(
    "VAULTLAYER_CHECKPOINT_DIR",
    os.environ.get(
        "CHECKPOINT_DIR",
        os.path.join(os.environ.get("VL_DATA_ROOT", "/tmp"), "vl-checkpoints", "mistral7b"),
    ),
)
os.makedirs(CKPT_DIR, exist_ok=True)
print(f"Checkpoints: {CKPT_DIR}")

resume_from = None
if os.path.isdir(CKPT_DIR):
    existing = sorted(
        [d for d in os.listdir(CKPT_DIR) if d.startswith("checkpoint-")],
        key=lambda x: int(x.split("-")[-1]) if x.split("-")[-1].isdigit() else 0,
    )
    if existing:
        resume_from = os.path.join(CKPT_DIR, existing[-1])
        print(f"Resuming from: {resume_from}")

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
    token=HF_TOKEN,
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = prepare_model_for_kbit_training(model)
print(f"Model loaded in {time.time() - t0:.0f}s")

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

print(f"\nLoading dataset: {DATASET_ID}...")
dataset = load_dataset(DATASET_ID, split="train")
print(f"Dataset: {len(dataset)} examples")


def format_alpaca(example):
    if example.get("input"):
        return f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"


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
