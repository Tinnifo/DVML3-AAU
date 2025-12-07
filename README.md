# Genome Representation Learning with K-mers


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



## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd P3_Final
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```


## Usage

1. Clone this repository:
```bash
git clone https://github.com/abdcelikkanat/revisitingkmers.git
cd revisitingkmers
```

2. Install dependencies: Make sure you have Python 3.8 installed. You can install the required Python packages using pip:

```bash
pip install -r requirements.txt
```

3. Install gdown (if you don't already have it) for downloading the datasets:

```bash
pip install gdown
Datasets
```
4. To download and prepare the training dataset, run the following commands:

```bash
gdown 1p59ch_MO-9DXh3LUIvorllPJGLEAwsUp
unzip dnabert-s_train.zip
```

5. To download the evaluation datasets, use the following commands:

```bash
gdown 1I44T2alXrtXPZrhkuca6QP3tFHxDW98c
unzip dnabert-s_eval.zip
```


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
