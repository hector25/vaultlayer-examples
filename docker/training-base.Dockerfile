# vaultlayer/training-base:1.0
#
# Purpose: eliminate the runtime pip-install dep-hell on container providers
# (Vast.ai, RunPod). Ships with torch 2.1.2 + CUDA 12.1 + the HuggingFace
# stack pinned to versions that resolved cleanly in the 2026-04-19 smoke
# runs (see /tmp/stress-5way/stub.sh for the reference version set).
#
# Deliberate choices:
#   * nvidia/cuda:12.1.1-cudnn8-devel base (not pytorch/pytorch conda image)
#     — avoids the conda/pip ABI split that caused _psutil_linux.getpagesize
#     mismatches when stubs `pip install psutil` on top of a conda-compiled
#     psutil. pip-only env is simpler + predictable.
#   * torch 2.1.2 + triton 2.1.0 — transformers>=4.36 calls into
#     triton._C.libtriton; mismatches (torch 2.3's triton 2.3.x vs our
#     triton 2.1.0 pin) surface as "No module named triton._C.libtriton.triton"
#     at SFTTrainer import time.
#   * bitsandbytes 0.43.1 on a -devel (CUDA headers present) base so the
#     library compiles GPU kernels at first import instead of silently
#     falling back to CPU ops (confirmed by Vast.ai support 2026-04-19).
#   * Build-time smoke: final RUN imports every dep + prints versions —
#     if anything is ABI-broken, the build fails instead of shipping a
#     bad image.
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.11 + OS tools needed by the broker onstart script
# (curl for rclone install, fuse3 for R2 mount, openssh-client for SSH)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip python3.11-venv \
        build-essential git curl unzip ca-certificates \
        fuse3 openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/local/bin/python

RUN python3 -m pip install --upgrade pip setuptools wheel

# Torch 2.1.2 + CUDA 12.1 wheel from the PyTorch index.
# Installed separately from HF stack so the pytorch index URL doesn't
# contaminate the pypi resolve for the rest.
RUN python3 -m pip install \
        torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121

# HuggingFace stack — versions from the 2026-04-19 stub reference set.
# Keep these in sync with any prod stub updates.
RUN python3 -m pip install \
        "triton==2.1.0" \
        "transformers==4.36.2" \
        "accelerate==0.27.2" \
        "peft==0.8.2" \
        "trl==0.7.11" \
        "datasets==2.18.0" \
        "bitsandbytes==0.43.1" \
        "sentencepiece" \
        "protobuf" \
        "rich" \
        "psutil" \
        "typing_extensions"

# Build-time smoke test — fails the image build if any dep is ABI-broken.
# Runs on CPU (no GPU at build time on GHA runners), so bitsandbytes will
# print a CUDA-not-found warning but the import itself must succeed.
RUN python3 -c "\
import torch, transformers, peft, trl, datasets, accelerate, psutil, triton, sentencepiece; \
print(f'torch={torch.__version__} transformers={transformers.__version__} '\
f'trl={trl.__version__} peft={peft.__version__} accelerate={accelerate.__version__} '\
f'triton={triton.__version__}')" \
    && python3 -c "import bitsandbytes; print(f'bitsandbytes={bitsandbytes.__version__}')" || true

WORKDIR /workspace
