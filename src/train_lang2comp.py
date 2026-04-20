"""
Train the Language-to-Composition network on synthetic data.

Usage:
    python src/train_lang2comp.py --epochs 50 --lr 1e-3 --device cuda:0
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lang2comp import Lang2Comp, generate_training_texts


class TextWeightDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['text'], torch.tensor(s['weights']), torch.tensor(s['params'])


def collate_fn(batch):
    texts, weights, params = zip(*batch)
    return list(texts), torch.stack(weights), torch.stack(params)


def train(args):
    print(f"Generating {args.n_samples} training samples...")
    all_samples = generate_training_texts(args.n_samples)

    # Split 90/10
    split = int(0.9 * len(all_samples))
    train_samples = all_samples[:split]
    val_samples = all_samples[split:]

    train_loader = DataLoader(
        TextWeightDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        TextWeightDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    print("Initializing Lang2Comp model...")
    model = Lang2Comp(hidden_dim=args.hidden_dim).to(args.device)

    # Only train the composition head (encoder is frozen)
    optimizer = torch.optim.AdamW(
        model.composition_head.parameters(),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        n_batches = 0

        for texts, target_w, target_p in train_loader:
            target_w = target_w.to(args.device)
            target_p = target_p.to(args.device)

            pred_w, pred_p = model(texts)

            # KL divergence for weights (distributions on simplex)
            loss_w = F.kl_div(pred_w.log(), target_w, reduction='batchmean')
            # MSE for params
            loss_p = F.mse_loss(pred_p, target_p)
            loss = loss_w + loss_p

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        scheduler.step()

        # Validate
        model.eval()
        val_loss = 0.0
        val_batches = 0
        correct_dominant = 0
        total = 0

        with torch.no_grad():
            for texts, target_w, target_p in val_loader:
                target_w = target_w.to(args.device)
                target_p = target_p.to(args.device)

                pred_w, pred_p = model(texts)

                loss_w = F.kl_div(pred_w.log(), target_w, reduction='batchmean')
                loss_p = F.mse_loss(pred_p, target_p)
                val_loss += (loss_w + loss_p).item()
                val_batches += 1

                # Dominant property accuracy
                pred_dom = pred_w.argmax(dim=1)
                target_dom = target_w.argmax(dim=1)
                correct_dominant += (pred_dom == target_dom).sum().item()
                total += len(texts)

        val_loss /= val_batches
        dom_acc = correct_dominant / total

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"DomAcc: {dom_acc:.2%} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_loss': val_loss,
                'dom_acc': dom_acc,
            }, 'checkpoints/lang2comp_best.pt')

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")

    # Test on examples
    model.eval()
    test_texts = [
        "a symmetric vase",
        "a smooth organic blob",
        "a compact cube",
        "a smooth and symmetric sculpture",
        "a rough asymmetric rock",
    ]
    print("\n--- Test Predictions ---")
    for text in test_texts:
        result = model.predict(text)
        print(f"  '{text}'")
        print(f"    weights: {result['weights']}")
        print(f"    dominant: {result['dominant_property']}")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-samples', type=int, default=20000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    train(args)
