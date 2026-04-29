
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


# Project Structure

```
Project Structure

LipRecognitionFYP/
├── data/
│   ├── grid/
│   └── processed/
│       ├── phonemes_by_label_mfa/
│       ├── phonemes_mfa/
│       ├── visemes_bozkurt_mfa/
│       ├── visemes_bozkurt_mfa_balanced_npy/
│       ├── words_by_label/
│       └── words_lip_bboxes.json
│
├── demo/
│   ├── lip_regions/
│   ├── phoneme_frames/
│   ├── phoneme_frames_mfa/
│   ├── phoneme_frames_mfa_2test/
│   ├── phoneme_frames_mfa_50/
│   ├── phoneme_jsons/
│   ├── plots/
│   ├── __init__.py
│   ├── alignment_manual_ratings.csv
│   ├── batch_equal_partition_results.json
│   ├── batch_mfa_2test_results.json
│   ├── demo_results.json
│   ├── evaluate_alignment_stats.py
│   ├── export_phoneme_frames_from_*.py
│   ├── extract_lip_regions_with_dlib.py
│   ├── extract_phonemes_equal_partition.py
│   ├── extract_words_from_grid_video.py
│   ├── lip_extraction_results.json
│   ├── mfa_phoneme_frame_extraction.py
│   ├── phoneme_extraction_equal_partition.json
│   ├── phoneme_extraction_mfa.json
│   ├── plot_alignment_stats.py
│   ├── run_demo_menu.py
│   ├── run_equal_partition_on_sample.py
│   ├── run_mfa_on_sample.py
│   ├── sample_50_grid_pairs.json
│   ├── sample_50_grid_videos.py
│   ├── save_phoneme_frames.py
│   ├── summarize_batch_mfa_results.py
│   └── summarize_equal_partition_results.py
│
├── initial_testing/
│   ├── simple_output/
│   └── initial_lip_detection.py
│
├── mapping/
│   ├── processed/
│   ├── bozkurt_viseme_map.csv
│   ├── organize_frames_by_viseme.py
│   └── test_mapping.py
│
├── preprocessing/
│   ├── dataset_splitting_utils.py
│   ├── precompute_lip_bboxes.py
│   ├── precompute_viseme_npy_dataset.py
│   ├── run_mfa_preprocessing_pipeline.py
│   ├── run_word_preprocessing_pipeline.py
│   └── split_dataset_train_val_test.py
│
├── testing/
│   ├── my_sounds/
│   ├── my_sounds_cropped_videos/
│   ├── my_words/
│   ├── my_words_cropped_videos/
│   ├── my_words_npy/
│   ├── demo_menu.py
│   ├── viseme_demo.py
│   └── webcam_test.py
│
├── tools/
│   ├── ffmpeg/
│   ├── preprocess_my_sounds.py
│   └── preprocess_my_words.py
│
├── training/
│   ├── checkpoints_words/
│   ├── results_bozkurt_viseme/
│   ├── results_words/
│   ├── runs_bozkurt_viseme/
│   ├── runs_words/
│   ├── dataset.py
│   ├── device.py
│   ├── infer.py
│   ├── model.py
│   ├── train_utils.py
│   ├── train_viseme_recognition_bozkurt_3d_cnn.py
│   └── train_word_recognition_3d_cnn.py
│
├── validation/
│   ├── results_bozkurt_viseme/
│   ├── results_comparison/
│   ├── results_viseme/
│   ├── results_words/
│   ├── compare_models_plot.py
│   ├── evaluate_model.py
│   ├── per_class_accuracy.py
│   └── qualitative_examples.py
│
├── visuals/
│   ├── viseme/
│   ├── viseme_eval/
│   ├── word/
│   ├── architecture_diagram.png
│   ├── architecture_diagram_viseme.png
│   ├── architecture_diagram_word.png
│   ├── data_statistics.png
│   ├── plot_architecture_diagram.py
│   ├── plot_viseme_class_analysis.py
│   ├── plot_word_training_results.py
│   ├── sample_video_blue.png
│   ├── visualize_viseme_data.py
│   └── visualize_words_model_and_data.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── shape_predictor_68_face_landmarks.dat.bz2
└── ...
```
