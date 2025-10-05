# 🎭 GFMamba: Multimodal Sentiment Analysis with Mamba-based Fusion

PyTorch implementation of multimodal sentiment analysis using the GFMamba model that integrates text, audio, and vision for datasets like CMU‑MOSI/CMU‑MOSEI.

---

## 🧠 Model Overview

GFMamba leverages Mamba-based sequence modeling for cross‑modal fusion:

- `Text`: language content
- `Audio`: prosody and tone
- `Vision`: facial expressions and motion

The default task is regression (predicting a scalar sentiment score per clip).

---

## ⚙️ Installation

Python 3.10+ is recommended.

Option A (conda, legacy env file in this repo):

```bash
conda env create -f require.yml
conda activate misa-code
```

Option B (conda manual + pip):

```bash
conda create -n gfmamba python=3.10 -y
conda activate gfmamba

# Install PyTorch matching your CUDA (example: CUDA 12.1)
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision torchaudio

# Core deps
pip install opencv-python pyyaml numpy

# Optional: microphone input for audio (uses PortAudio)
pip install sounddevice
```

Notes:

- If you plan to train, set `configs/mosi_train.yaml: dataset.dataPath` to your local `aligned_50.pkl`.
- Checkpoint files are expected under `ckpt/<dataset>/` (e.g., `ckpt/mosi/`).

---

## 🎥 Real‑Time Inference (Webcam + Mic)

Run with the provided config and checkpoint (GPU):

```bash
python app/realtime_infer.py \
  --config configs/mosi_train.yaml \
  --ckpt ckpt/mosi/best_valid_model_seed_42.pth \
  --device cuda \
  --camera -1
```

Tips:

- Camera backend on Windows defaults to DirectShow. If it fails, try:

```bash
python app/realtime_infer.py --config configs/mosi_train.yaml --ckpt ckpt/mosi/best_valid_model_seed_42.pth --device cuda --camera -1 --backend msmf
```

- List available cameras:

```bash
python app/realtime_infer.py --list-cameras
```

- CPU mode:

```bash
python app/realtime_infer.py --device cpu --camera -1
```

During load you may see messages like “missing/unexpected keys” if the checkpoint was created with a slightly different code version. Loading is non‑strict and the app will still run; accuracy may differ.

---

## 📚 Training (optional)

Edit the config first (notably `dataset.dataPath`), then run:

```bash
python train.py --config_file configs/mosi_train.yaml --seed 42
```

---

If you are interested in this project, contact: `zzhe232@aucklanduni.ac.nz`.
