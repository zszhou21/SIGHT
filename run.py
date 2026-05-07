#!/usr/bin/env python3
"""SIGHT runner: source training or test-time adaptation.

Usage:
    python run.py --phase source_training --data data/harth --source S006 --num-classes 5
    python run.py --phase tta --data data/harth --source S006 --target S032 --ckpt outputs/source_cnn.pt --num-classes 5
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn

from src.cnn import SimpleHARCNN
from src.SIGHT import SIGHT


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_npz(path, labels):
    blob = np.load(path, allow_pickle=True)
    idx = {lbl: i for i, lbl in enumerate(labels)}
    x = blob["x"]
    y = np.array([idx[str(l)] for l in blob["y"].tolist()], dtype=np.int64)
    return x, y


def train_source(args):
    data_dir = Path(args.data)
    x, y = load_npz(data_dir / f"{args.source}.npz", args.labels)
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(x).float(), torch.from_numpy(y).long())
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=len(dataset) > args.batch_size)

    model = SimpleHARCNN(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        mid_channels=args.mid_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        final_out_channels=args.final_out_channels,
        features_len=args.features_len,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            optimizer.step()

    save_path = Path(args.ckpt)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, save_path)
    print(f"Saved source model: {save_path}")


def run_tta(args):
    data_dir = Path(args.data)
    x, y = load_npz(data_dir / f"{args.target}.npz", args.labels)

    model = SimpleHARCNN(
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        mid_channels=args.mid_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        final_out_channels=args.final_out_channels,
        features_len=args.features_len,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(args.device).eval()

    so_preds = []
    with torch.no_grad():
        for i in range(len(x)):
            xb = torch.from_numpy(x[i]).unsqueeze(0).to(args.device, torch.float32)
            so_preds.append(model(xb).argmax(dim=1).item())
    so_f1 = round(f1_score(y, so_preds, average="macro"), 4)
    so_acc = round(accuracy_score(y, so_preds), 4)
    print(f"Source-only  F1={so_f1:.4f}  Acc={so_acc:.4f}")

    sight = SIGHT(
        model, args.device, args.num_classes,
        eta_mu=args.eta_mu, omega_mu=args.omega_mu,
        beta=args.beta, eta_h=args.eta_h, tau=args.tau,
    )
    tta_preds = []
    for i in range(len(x)):
        xb = torch.from_numpy(x[i]).unsqueeze(0).to(args.device, torch.float32)
        q = sight.step(xb)
        tta_preds.append(q.argmax(dim=1).item())
    tta_f1 = round(f1_score(y, tta_preds, average="macro"), 4)
    tta_acc = round(accuracy_score(y, tta_preds), 4)
    print(f"SIGHT (ours) F1={tta_f1:.4f}  Acc={tta_acc:.4f}  (ΔF1={tta_f1 - so_f1:+.4f})")

    if args.output:
        result = {
            "source": args.source,
            "target": args.target,
            "source_only": {"accuracy": so_acc, "macro_f1": so_f1},
            "sight": {"accuracy": tta_acc, "macro_f1": tta_f1, "delta": round(tta_f1 - so_f1, 4)},
        }
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Saved result: {args.output}")


def main():
    parser = argparse.ArgumentParser(description="SIGHT")
    parser.add_argument("--phase", required=True, choices=["source_training", "tta"])
    parser.add_argument("--data", required=True, help="Path to data directory containing .npz files")
    parser.add_argument("--source", required=True, help="Source subject ID")
    parser.add_argument("--target", default=None, help="Target subject ID (required for tta)")
    parser.add_argument("--ckpt", default="outputs/source_cnn.pt", help="Checkpoint path")
    parser.add_argument("--output", default=None, help="Optional JSON output path for tta results")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=42)

    # Model architecture
    parser.add_argument("--num-classes", type=int, default=5)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--mid-channels", type=int, default=64)
    parser.add_argument("--kernel-size", type=int, default=5)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--final-out-channels", type=int, default=128)
    parser.add_argument("--features-len", type=int, default=16)

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)

    # SIGHT hyperparameters
    parser.add_argument("--eta-mu", type=float, default=0.005, help="Prototype evolution rate")
    parser.add_argument("--omega-mu", type=float, default=0.01, help="Source-prototype anchor rate")
    parser.add_argument("--beta", type=float, default=1.0, help="Surprise sensitivity")
    parser.add_argument("--eta-h", type=float, default=0.05, help="Habit-vector learning rate")
    parser.add_argument("--tau", type=float, default=0.05, help="KVA temperature")

    # Labels (comma-separated)
    parser.add_argument("--labels", type=str, default="cycling_like,lying,sitting,standing,walking_like",
                        help="Comma-separated class labels")

    args = parser.parse_args()
    args.labels = [l.strip() for l in args.labels.split(",")]
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    if args.phase == "source_training":
        train_source(args)
    elif args.phase == "tta":
        if not args.target:
            parser.error("--target is required for tta phase")
        run_tta(args)


if __name__ == "__main__":
    main()
