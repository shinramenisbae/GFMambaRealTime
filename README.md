# 🎭 GFMamba: Multimodal Sentiment Analysis with Mamba-based Fusion

A PyTorch-based implementation of **multimodal sentiment analysis**, using the **GFMamba** model that integrates **text**, **audio**, and **vision** features for tasks like CMU-MOSI and CMU-MOSEI.

---

## 🧠 Model Overview

**GFMamba** is a multimodal fusion model that leverages the Mamba architecture to integrate and learn from:

- `Text` (language content)
- `Audio` (voice tone & prosody)
- `Vision` (facial expressions & motion)

It performs **regression-based sentiment prediction**, aiming to predict a sentiment score per utterance or clip.
datasets use cmu-mosi, if you want to transfer other datasets, you can change the yaml and datasets floder

---

## 🛠️ Installation

### 🔹 Clone and install dependencies

```bash
git clone https://github.com/yourusername/gfmamba.git
cd gfmamba
pip install -r requirements.txt

## requirement

```bash
pip install pytorch>2.1 python>3.9

if you interested in this project,concat us by email zzhe232@aucklanduni.ac.nz
