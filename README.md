# Human Emotion Detection System

A state-of-the-art multimodal emotion recognition pipeline designed to detect human emotions from video, audio, and text inputs. This system processes complex multimodal data streams, fuses them intelligently, and predicts emotional states with high accuracy.

##  Overview

This repository contains an end-to-end Machine Learning pipeline that trains, tunes, and exports a multimodal emotion detection model. Built with PyTorch and HuggingFace Transformers, the system leverages large language models (LLMs) alongside robust vision and audio encoders to achieve comprehensive contextual understanding of human expressions.

## Architecture (Multimodal Fusion)

The core architecture (`pipeline/model.py`) relies on a robust multimodal fusion strategy:

1. **Text Modality (LLM):** Utilizes **Llama-3.2-3B** (4-bit quantized) with **LoRA** (Low-Rank Adaptation) for highly contextualized text-based feature extraction.
2. **Audio Modality:** Employs a **CNN-BiLSTM** architecture processing Mel-spectrograms and MFCC audio features.
3. **Video Modality:** Uses an **r3d_18** (3D ResNet) backbone combined with **DynE** (Dynamic Encoder for temporal self-attention) to extract frame-level spatial-temporal features.
4. **Multimodal Fusion (TA-AVN):** A Text-Attended Audio-Visual Network where text tokens dynamically attend to concatenated audio and video representations.
5. **Reasoning & Classification:** Refined via **MER-ML(SE)** (Squeeze-Excitation gating) and a 4-Layer Transformer **Reasoner**, outputting the final emotion classification probabilities.

##  Project Structure

```text
├── models/             # Downloaded backbone weights (Llama-3.2-3B, r3d_18)
├── pipeline/           # Core ML execution pipeline
│   ├── 00_preflight.py # System & resource validation
│   ├── 01_diagnostics.py # Data integrity checks
│   ├── 02_tune.py      # Optuna hyperparameter tuning
│   ├── 03_train_full.py# Main training loop
│   ├── 04_export.py    # Model export for deployment
│   ├── model.py        # Complete neural network architecture
│   └── run_all.sh      # Master script to run the full pipeline
├── launch/             # Cluster & HPC launch scripts (PBS/tmux)
│   ├── submit.pbs      # Submit job to compute nodes
│   └── interactive.sh  # Run interactively via tmux
├── scripts/            # Helper scripts (model downloads, etc.)
└── runs/               # Checkpoints, logs, and exported bundles
```

##  Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Princeyadav774623/Human-Emotion-Detection-System.git
   cd Human-Emotion-Detection-System
   ```

2. **Set up the virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Download Model Backbones:**
   Make sure you have a HuggingFace Token configured for Llama-3.2:
   ```bash
   export HF_TOKEN="hf_your_token_here"
   bash scripts/download_models.sh
   ```

##  Running the Pipeline

The project provides a fully automated pipeline (`pipeline/run_all.sh`) that takes care of pre-flight checks, tuning, training, and exporting. 

**Option 1: Run locally**
```bash
bash pipeline/run_all.sh
```

**Option 2: Run on an HPC/Cluster (Recommended for heavy training)**
```bash
qsub launch/submit.pbs
# Check the launch/README.md for more HPC details.
```

##  Output & Export

Once Stage 4 (`04_export.py`) is complete, a fully standalone deployment bundle is created in `runs/<timestamp>/deploy/`. This bundle contains the optimized model weights, tokenizers, and configuration files ready for production deployment.
