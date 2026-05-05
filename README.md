# SIGHT

Temporal Structure Matters for Efficient Test-Time Adaptation in Wearable Human Activity Recognition.

## Requirements

```bash
pip install torch numpy scikit-learn
```

## Data

Download preprocessed `.npz` files from [[Google Drive - SIGHT-preprocessed-data]](https://drive.google.com/drive/folders/1tqDrggsd89mJ8FmDKNvipRWYhP9D13WN?usp=drive_link) and place them under `data/`.

Each `.npz` contains `x` (windows, channels, length) and `y` (string labels).

## Run

**Source training**
```bash
python run.py --phase source_training --data data/harth --source S006 --num-classes 5 --labels "cycling_like,lying,sitting,standing,walking_like"
```

**Test-time adaptation**
```bash
python run.py --phase tta --data data/harth --source S006 --target S032 --ckpt outputs/source_cnn.pt --num-classes 5
```

## Config

All hyperparameters are set via `argparse` in `run.py`:

| Arg | Default | Description |
|---|---|---|
| `--eta-mu` | 0.01 | Prototype evolution rate |
| `--beta` | 1.0 | Surprise sensitivity |
| `--eta-h` | 0.05 | Habit-vector learning rate |
| `--tau` | 0.2 | KVA temperature |
| `--epochs` | 100 | Source training epochs |
| `--lr` | 0.001 | Source training learning rate |
| `--batch-size` | 64 | Source training batch size |

Model architecture (channels, kernel sizes, etc.) is also configurable via CLI args.
