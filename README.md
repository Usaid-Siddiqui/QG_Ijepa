# QG-IJEPA: Self-Supervised Quark/Gluon Jet Tagging with I-JEPA

A PyTorch implementation of [I-JEPA](https://arxiv.org/abs/2301.08243) (Image-based Joint-Embedding Predictive Architecture) applied to quark/gluon jet discrimination in high-energy physics. A Vision Transformer is pretrained on jet images in a self-supervised manner — no labels required — then fine-tuned with a small labeled dataset for binary classification.

---

## How It Works

I-JEPA trains a ViT encoder by predicting the latent representations of masked target patches from visible context patches. A student (context encoder + predictor) learns to predict what a teacher (EMA-updated target encoder) would output for the masked regions. This forces the model to learn rich, semantic representations of jet substructure without any supervision.

```
Context Patches → Context Encoder → Predictor → Predicted Latents
                                                        ↕ MSE Loss
Target Patches  → Target Encoder (EMA) ─────────→ Target Latents
```

After pretraining, the frozen encoder is used as a feature extractor, and a lightweight MLP head is fine-tuned for quark vs. gluon classification.

---

## Project Structure

```
.
├── models/
│   ├── __init__.py          # Model exports
│   ├── vit.py               # Vision Transformer (encoder)
│   ├── predictor.py         # MaskPredictor (student predictor head)
│   └── ijepa.py             # IJEPA wrapper (context encoder + EMA target encoder)
├── train_pretrain.py        # Self-supervised pretraining loop
├── train_finetune.py        # Supervised fine-tuning with MLP probe
├── train_vit_comparison.py  # Side-by-side comparison: pretrained vs. scratch
├── check_collapse.py        # Latent space collapse diagnostic
├── config.yaml              # Main config (local cluster)
├── colab_config.yaml        # Config for Google Colab
├── QuarkGluon.ipynb         # End-to-end notebook
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA-capable GPU recommended.

```bash
pip install -r requirements.txt
```

Data should be in HDF5 format. Update the paths in `config.yaml` to point to your `train`, `finetune`, and `test` splits.

---

## Usage

### 1. Pretraining (Self-Supervised)

```bash
python train_pretrain.py
```

Trains the I-JEPA model on unlabeled jet images. Checkpoints are saved to `train.checkpoint_dir` in `config.yaml`. The best encoder weights are saved as `best_encoder.pth`.

Key config options under `train:`:

| Key | Description |
|---|---|
| `base_lr` | Peak learning rate (cosine schedule with warmup) |
| `epochs` | Total pretraining epochs |
| `warmup_epochs` | Linear LR warmup duration |
| `ema_momentum` | EMA decay for the target encoder (e.g. `0.999`) |
| `weight_decay` | AdamW weight decay |
| `save_freq` | Checkpoint every N epochs |

Masking strategy is controlled under `masking:` — adjust `num_targets`, `target_scale`, and `context_scale` to change how aggressively patches are masked.

### 2. Fine-tuning

```bash
python train_finetune.py
```

Loads a pretrained encoder and trains an MLP classification head on labeled data. Outputs a ROC curve and AUC score to `finetune.finetune_dir`.

Key config options under `finetune:`:

| Key | Description |
|---|---|
| `checkpoint_to_load` | Path to `best_encoder.pth` from pretraining |
| `freeze_encoder` | If `true`, only the MLP head is trained |
| `unfreeze_last` | Unfreeze the last N transformer blocks |
| `head_layers` | Hidden layer sizes for the MLP, e.g. `[512]` |
| `pool` | Patch aggregation: `mean`, `max`, or `cls` |
| `data_fraction` | Use a fraction of training data (e.g. `0.1` for low-label regime) |

### 3. Pretrained vs. Scratch Comparison

```bash
python train_vit_comparison.py
```

Runs two full fine-tuning jobs back-to-back — one with the pretrained encoder, one initialized from scratch — using identical hyperparameters. Produces a combined plot with three panels:

- ROC curves for both runs
- Validation AUC vs. epoch
- Compute efficiency: AUC vs. total wall-clock time (if `pretrain_time_min` is set in config)

This is the main tool for quantifying the value of self-supervised pretraining. Configure under `vit_comparison:` in `config.yaml`.

---

## Monitoring Training Health

### Checking for Representation Collapse

If training loss stagnates or performance is poor, check whether the encoder has collapsed to a trivial constant embedding:

```python
# check_collapse.py
python check_collapse.py
```

An average latent standard deviation below `0.01` indicates collapse. Healthy encoders typically show std > `0.1`.

### Signs of Healthy Pretraining

- Loss decreases steadily over the first ~10 epochs
- Loss does not plateau near zero (would indicate target encoder moving too fast)
- `check_collapse.py` reports diverse latent representations

---

## Architecture

**Vision Transformer (ViT-Small)**

| Parameter | Value |
|---|---|
| Patch size | 8×8 pixels |
| Patch embedding dim | 192 |
| Encoder embedding dim | 384 |
| Depth | 12 blocks |
| Attention heads | 6 |
| Positional encoding | Fixed 2D sine-cosine |

**Predictor**

| Parameter | Value |
|---|---|
| Input/output dim | 384 (encoder dim) |
| Predictor dim | 192 |
| Depth | 6 blocks |

The predictor takes context latents + masked target positions (via positional embeddings) and predicts the target encoder's output at those positions.

---

## Results

Fine-tuning the pretrained encoder with a frozen backbone achieves **AUC ≈ 0.707** after 4 epochs of training on the full labeled set. This baseline can be improved by unfreezing the encoder (`unfreeze_last`) or extending training.

---

## Running on Google Colab

Use `colab_config.yaml` (set as the config path in the notebook) and `QuarkGluon.ipynb`. The notebook handles data decompression, pretraining, and fine-tuning end-to-end. Checkpoints are persisted to Google Drive.

---

## Citation

This project applies the I-JEPA framework from:

> Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*, CVPR 2023. ([arXiv:2301.08243](https://arxiv.org/abs/2301.08243))

---

*By Usaid Siddiqui*