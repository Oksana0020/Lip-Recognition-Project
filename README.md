# Training

The two lip-reading models built for this project: a **word-level model** and a **viseme-level model**.

---

## What Was Built

Two separate 3D CNN models were trained on the GRID corpus,a dataset of controlled speech videos where speakers read out short sentences in front of a camera.

| | Word Model | Viseme Model |
|---|---|---|
| **What it predicts** | Which word was spoken (51 classes) | Which mouth shape group was shown (13 classes) |
| **Training data** | `data/processed/words_by_label/` | `data/processed/visemes_bozkurt_mfa_balanced_npy/` |
| **Samples** | 59,370 clips | 65,035 clips |
| **Best checkpoint** | `checkpoints_words/best_model_words.pth` | `checkpoints_bozkurt_viseme/bozkurt_viseme_best_model.pth` |
| **Val accuracy** | **88.20%** | **82.78%** |
| **Best epoch** | 54 / 60 | 60 / 60 |

---

## Results

### Word Model
- Validation accuracy: **88.20%**
- Full-dataset accuracy: **96.10%**
- Macro F1 score: **0.943**
- Top word classes: *please* 99.6%, *zero* 99.2%, *four* 99.4%

### Viseme Model
- Validation accuracy: **82.78%**
- Average per-class accuracy: **~91.3%** across 13 classes
- Easiest class: **V8** (diphthong /aw/) — 99.0%
- Hardest class: **V13** (post-alveolar /ch, sh, jh, zh/) — 79.1%
- Lip-visible classes average ~94%, tongue/internal classes average ~88%

---

## Model Architecture

Both models share the exact same architecture — only the number of output classes differs.

```
Input:  [batch, 1 channel, 8 frames, 64×64 pixels]  ← grayscale lip crop

Block 1:  Conv3D(1→32)   → BatchNorm → ReLU → MaxPool(1,2,2)
Block 2:  Conv3D(32→64)  → BatchNorm → ReLU → MaxPool(2,2,2)
Block 3:  Conv3D(64→128) → BatchNorm → ReLU → MaxPool(2,2,2)
Block 4:  Conv3D(128→256)→ BatchNorm → ReLU → AdaptiveAvgPool

Flatten → Dense(4096→512) → ReLU → Dropout(0.5) → Dense(512→N classes)

Output: [batch, N]   ← N=51 for words, N=13 for visemes
```

**Total parameters: ~1.3 million**

---

## Scripts

### Training
| Script | What it does |
|---|---|
| `train_word_recognition_3d_cnn.py` | Trains the word model from scratch |
| `train_viseme_recognition_bozkurt_3d_cnn.py` | Trains the viseme model from scratch |

### Inference (running the trained model on a clip)
| Script | What it does |
|---|---|
| `inference_word_recognition.py` | Predicts which word is shown in a `.npy` clip |
| `inference_viseme_bozkurt.py` | Predicts which viseme class is shown in a `.npy` clip |
---

## How to Run Inference

**Predict a word from a clip:**
```bash
python training/inference_word_recognition.py \
  --video_path data/processed/words_by_label/please/bbaczp_word_39750_please.npy \
  --top_k 3
```

**Predict a viseme from a clip:**
```bash
python training/inference_viseme_bozkurt.py \
  --video data/processed/visemes_bozkurt_mfa_balanced_npy/V8/some_clip.npy \
  --checkpoint training/checkpoints_bozkurt_viseme/bozkurt_viseme_best_model.pth
```

---

## How to Re-train

Make sure data is preprocessed first (see `preprocessing/`), then:

```bash
# Word model
python training/train_word_recognition_3d_cnn.py

# Viseme model
python training/train_viseme_recognition_bozkurt_3d_cnn.py
```

Training saves:
- Best model checkpoint (by validation accuracy)
- A checkpoint every 5 epochs
- TensorBoard logs in `runs_words/` or `runs_bozkurt_viseme/`

Monitor training live:
```bash
tensorboard --logdir=training/runs_words
# then open http://localhost:6006
```

---

## Training Configuration

Both scripts use a `TrainingConfig` class at the top of the file. Key settings:

| Setting | Word Model | Viseme Model |
|---|---|---|
| Frames per clip | 8 | 8 |
| Frame size | 64×64 | 64×64 |
| Colour | Grayscale (1 channel) | Grayscale (1 channel) |
| Batch size | 32 | 32 |
| Epochs | 60 | 60 |
| Learning rate | 0.001 | 0.001 |
| Train/val/test split | 70/15/15% | 70/15/15% |
| Weighted sampler | Yes (handles class imbalance) | Yes |

---

## Viseme Classes (Bozkurt System)

The viseme model uses Bozkurt phoneme-to-viseme mapping, which groups phonemes by how similar they look on lips

| Viseme | Phonemes | Mouth Shape | Val Accuracy |
|---|---|---|---|
| V2 | ay, ah | Open mouth | 96.8% |
| V3 | ey, eh, ae | Mid-open | 91.6% |
| V6 | uw, uh, w | Rounded lips | 95.3% |
| V7 | ao, aa, oy, ow | Back rounded | 97.4% |
| V8 | aw | Diphthong | **99.0%** |
| V9 | g, hh, k, ng | Velar (back of mouth) | 89.1% |
| V10 | r | R-coloured | 92.1% |
| V11 | l, d, n, t, en, el | Alveolar (tongue tip) | 89.1% |
| V12 | s, z | Alveolar fricatives | 91.7% |
| V13 | ch, sh, jh, zh | Post-alveolar | **79.1%** |
| V14 | th, dh | Dental fricatives | 84.7% |
| V15 | f, v | Labiodental | 92.0% |
| V16 | m, em, b, p | Bilabial (lips together) | 89.4% |

V8 is the easiest because the diphthong /aw/ has a very distinctive lip movement.
V13 is the hardest because post-alveolar sounds look similar to each other and have fewer training samples.

---

