# MMSD

Official implementation of **MMSD: A Multimodal Multitask Self-Distillation Framework for Audio-Text Emotion Recognition**.

## Structure

```text
MMSD-main/
  iemocap_code/
    stage1/        # Stage-I multimodal multitask backbone
    stage2/        # Stage-II self-distillation
  meld_code/
    stage1/
    stage2/
```

## Requirements

The code was developed with Python 3.8 and PyTorch. Main dependencies:

- torch
- torchaudio
- transformers
- pandas
- numpy
- scikit-learn
- tqdm
- soundfile

## IEMOCAP

Run Stage-I training for one fold:

```bash
cd iemocap_code/stage1
python train.py --fold 1 --seed 42
```

Generate fold-specific soft labels:

```bash
cd iemocap_code/stage2
python generate_soft_labels.py \
  --models_dir checkpoints/iemocap_stage1 \
  --csv_path feats/train_augmented_clean.csv \
  --audio_dir data/aug_wav \
  --output_dir soft_labels_dir
```

Run Stage-II self-distillation:

```bash
cd iemocap_code/stage2
python train_soft.py \
  --fold 1 \
  --teacher_dir checkpoints/iemocap_stage1 \
  --hard_csv feats/train2.csv \
  --soft_csv feats/train_augmented_clean.csv \
  --soft_labels_dir soft_labels_dir
```

## MELD

Run Stage-I training:

```bash
cd meld_code/stage1
python train.py
```

Generate soft labels:

```bash
cd meld_code/stage2
python generate_soft_labels.py \
  --model_path checkpoints/meld_stage1/meld_best_model.pt \
  --csv_path data/aug_meld_processed/train/meld_train_augmented.csv \
  --audio_dir data/aug_meld_processed/train/wavs \
  --output_dir meld_soft_labels_dir
```

Run Stage-II self-distillation:

```bash
cd meld_code/stage2
python train_soft.py \
  --teacher_model checkpoints/meld_stage1/meld_best_model.pt \
  --train_csv_hard data/meld_processed/train/meld_train.csv \
  --train_csv_soft data/aug_meld_processed/train/meld_train_augmented.csv \
  --train_audio_hard data/meld_processed/train/wavs \
  --train_audio_soft data/aug_meld_processed/train/wavs
```

Dataset files and checkpoints are not included. Please prepare IEMOCAP and MELD according to their official licenses and pass the corresponding paths through command-line arguments when your local layout differs from the defaults above.
