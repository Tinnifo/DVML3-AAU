# Genome Representation Learning with K-mers

This repository contains implementations of two models for genome representation learning:
- **REVISIT**: A baseline model using negative sampling and binary cross-entropy loss
- **OUR**: A supervised contrastive learning (SupCon) model with multi-view augmentation

## Overview

Both models learn embeddings from DNA sequences by:
1. Converting sequences to k-mer profiles
2. Encoding k-mer profiles through a neural network
3. Learning representations that capture sequence similarity

The **OUR** model extends the baseline by:
- Using multi-view augmentation (sliding windows + reverse complements)
- Applying supervised contrastive learning to pull together views from the same fragment
- Better handling of long sequences through view splitting

## Installation

### Requirements

- Python 3.7+
- CUDA-capable GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd P3_Final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
P3_Final/
├── src/
│   ├── OUR.py              # SupCon model implementation
│   └── REVISIT.py          # Baseline model implementation
├── sh/
│   ├── train_our.sh        # Training script for OUR model
│   ├── train_revisit.sh    # Training script for REVISIT model
│   └── evaluate_models.sh  # Evaluation script
├── utils/
│   └── utils.py            # Utility functions (embedding, clustering)
├── binning.py              # Evaluation script for binning tasks
├── train_2m.csv            # Training data (read pairs)
├── val_48k.csv             # Validation data
├── reference/              # Reference species data
├── marine/                 # Marine species data
├── plant/                  # Plant species data
└── requirements.txt        # Python dependencies
```

## Usage

### Training Models

#### Train OUR (SupCon) Model

```bash
./sh/train_our.sh
```

The script uses the following default parameters:
- `k=4` (k-mer size)
- `dim=256` (embedding dimension)
- `epoch=2` (number of epochs)
- `lr=0.001` (learning rate)
- `batch_size=10000`
- `max_read_num=100`
- `temperature=0.1` (SupCon temperature)

To customize, edit the variables at the top of `sh/train_our.sh`.

#### Train REVISIT Model

```bash
./sh/train_revisit.sh
```

Default parameters:
- `k=4`
- `dim=256`
- `epoch=1000`
- `lr=0.001`
- `batch_size=10000`
- `neg_sample_per_pos=1000`
- `loss_name=bern` (bern, poisson, or hinge)

### Evaluation

Evaluate both models on binning tasks:

```bash
./sh/evaluate_models.sh
```

Make sure to update the model paths in `sh/evaluate_models.sh` to point to your trained models:
```bash
REVISIT_MODEL=${BASEFOLDER}/models/revisit_k4_dim256_epoch=1000_LR0.001_batch=10000.pt
OUR_MODEL=${BASEFOLDER}/models/our_k4_dim256_epoch=2_LR0.001_batch=10000.pt
```

The evaluation script will:
- Generate embeddings for test sequences
- Run KMedoid clustering
- Compute F1 scores and recall metrics
- Save results to `results/evaluation_*.txt`

## Model Details

### REVISIT Model

- **Architecture**: Linear → BatchNorm → Sigmoid → Dropout → Linear
- **Loss**: Binary cross-entropy (or Poisson/Hinge)
- **Training**: Negative sampling with paired reads

### OUR Model

- **Architecture**: Linear → BatchNorm → ReLU → Dropout → Linear
- **Loss**: Supervised Contrastive Loss
- **Training**: Multi-view augmentation with SupCon
- **View Generation**:
  - Splits long sequences into overlapping windows
  - Generates reverse complements
  - Creates multiple views per read pair

## Data Format

Training data should be in CSV format with two columns:
```
left_read,right_read
ATCGATCG...,GCTAGCTA...
...
```

Each line contains a pair of DNA sequences (left and right reads from the same fragment).

## Output

### Training Output

- **Model file**: `models/{model_name}_k{K}_dim{DIM}_epoch={EPOCH}_LR={LR}_batch={BATCH}.pt`
- **Loss log**: TensorBoard logs (view with `tensorboard --logdir models/`)

### Evaluation Output

- **Results file**: `results/evaluation_{TIMESTAMP}.txt`
  - Contains F1 scores, recall results, and thresholds for each model/species/sample

## Parameters

### Key Hyperparameters

- **k**: k-mer size (default: 4)
- **dim**: Embedding dimension (default: 256)
- **epoch**: Number of training epochs
- **lr**: Learning rate (default: 0.001)
- **batch_size**: Batch size (default: 10000)
- **temperature**: SupCon temperature parameter (OUR model only, default: 0.1)
- **max_views_per_read**: Maximum views generated per read (OUR model only, default: 4)

### View Splitting Parameters (OUR model)

Automatically calculated based on `k`:
- `L_min_useful = 4 * k²` (minimum useful length)
- `W_target = 4 * L_min_useful` (target window size)

## GPU Usage

To use GPU, set `DEVICE=cuda` in the training scripts:

```bash
# Edit sh/train_our.sh and change:
DEVICE=cuda
```

Or modify the script directly before running.

## Citation

If you use this code, please cite the original papers:
- REVISIT baseline model
- Supervised Contrastive Learning for genomic sequences

## License

[Add your license here]

## Contact

[Add contact information here]

# Self-supervised-Contrastive-Learning-for-Genomic-Sequence-Embedding
# Self-supervised-Contrastive-Learning-for-Genomic-Sequence-Embedding
# x
