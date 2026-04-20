"""Retrain Lang2Comp with Pareto-aware (flattened) training targets.

Root cause of underperformance (see FINAL_EXPERIMENT_REPORT.md §3.9):
  Current training templates assign peaked one-hot-like labels
  (e.g. sym_high -> [0.8, 0.1, 0.1]). After training, the MLP predicts
  weights with max ~0.7 on 75% of test prompts, pushing outputs into the
  extreme corners of the simplex where the Pareto cloud is thin.

Measured Pareto cloud on 97 test prompts lives near [0.4, 0.6] on the
dominant axis. Equal Wt (1/3,1/3,1/3) beats pure Lang2Comp by +14.2 pp
symmetry at alpha=1.0 in the blend sweep.

Fix (this script):
  Shrink every synthetic weight target toward the simplex centroid:
      w_new = (1-lam) * w_original + lam * [1/3, 1/3, 1/3]
  with lam = 0.5. This:
    - keeps the language signal (dominant axis is still highest)
    - caps the peak at ~0.57 instead of 0.8
    - moves targets INTO the Pareto cloud region

Checkpoint saved to checkpoints/lang2comp_v2.pt (does NOT overwrite the
existing lang2comp_retrained.pt).
"""
import argparse
import json
import os
import random
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from lang2comp import Lang2Comp, TRAINING_TEMPLATES, OBJECTS


def _shrink(w, lam):
    """Blend toward simplex centroid: w_new = (1-lam)*w + lam * [1/3, 1/3, 1/3]."""
    c = 1.0 / 3.0
    return [(1 - lam) * wi + lam * c for wi in w]


def generate_training_texts_v2(n_samples, lam):
    """Same categories / templates as original but with flattened weight targets.

    The original weight_map peaks at 0.8. After shrinkage with lam=0.5 the peak
    becomes 0.567, which is inside the observed Pareto cloud.
    """
    weight_map = {
        'symmetry_high': ([0.8, 0.1, 0.1], [0.9, 0.5, 0.5]),
        'symmetry_low':  ([0.8, 0.1, 0.1], [0.1, 0.5, 0.5]),
        'smoothness_high': ([0.1, 0.8, 0.1], [0.5, 0.9, 0.5]),
        'smoothness_low':  ([0.1, 0.8, 0.1], [0.5, 0.1, 0.5]),
        'compactness_high': ([0.1, 0.1, 0.8], [0.5, 0.5, 0.9]),
        'compactness_low':  ([0.1, 0.1, 0.8], [0.5, 0.5, 0.1]),
        'composite': ([1/3, 1/3, 1/3], [0.7, 0.7, 0.7]),
    }
    # Pre-shrink every category EXCEPT 'composite' (already centroid).
    shrunk_map = {}
    for cat, (w, p) in weight_map.items():
        shrunk_map[cat] = (_shrink(w, lam), p)

    samples = []
    for _ in range(n_samples):
        category = random.choice(list(TRAINING_TEMPLATES.keys()))
        template = random.choice(TRAINING_TEMPLATES[category])
        obj = random.choice(OBJECTS)
        text = template.format(object=obj)

        w, p = shrunk_map[category]
        # Small Gaussian jitter for diversity, same as original.
        w_noisy = [max(0.01, wi + random.gauss(0, 0.03)) for wi in w]
        p_noisy = [max(0.01, min(0.99, pi + random.gauss(0, 0.05))) for pi in p]
        s = sum(w_noisy)
        w_noisy = [wi / s for wi in w_noisy]
        samples.append({'text': text, 'weights': w_noisy, 'params': p_noisy})
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, default=0.5,
                    help='shrinkage toward [1/3, 1/3, 1/3] centroid (0=original, 1=pure equal wt)')
    ap.add_argument('--n-samples', type=int, default=10000)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', type=str, default='cpu')
    ap.add_argument('--out', type=str, default='checkpoints/lang2comp_v2.pt')
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f'[v2] lam={args.lam}  n_samples={args.n_samples}  epochs={args.epochs}  device={args.device}')
    print(f'[v2] Generating {args.n_samples} synthetic samples with flattened targets...')
    samples = generate_training_texts_v2(args.n_samples, args.lam)
    texts = [s['text'] for s in samples]
    labels = torch.tensor([s['weights'] for s in samples], dtype=torch.float32)

    print(f'[v2] Label stats:')
    print(f'     max weight mean: {labels.max(dim=1).values.mean():.3f}')
    print(f'     min weight mean: {labels.min(dim=1).values.mean():.3f}')
    print(f'     mean weight   : {labels.mean(dim=0).tolist()}')

    model = Lang2Comp()
    model.to(args.device)

    print('[v2] Encoding texts (once, frozen encoder)...')
    with torch.no_grad():
        embeddings = model.encode_text(texts)

    dataset = TensorDataset(embeddings, labels.to(args.device))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    optimizer = torch.optim.Adam(model.composition_head.parameters(),
                                 lr=args.lr, weight_decay=1e-4)

    print(f'[v2] Training {args.epochs} epochs (MSE loss, matches tools/train_lang2comp.py)...')
    best_loss = float('inf')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for emb_batch, label_batch in loader:
            optimizer.zero_grad()
            logits = model.composition_head(emb_batch)
            w_pred = torch.softmax(logits[:, :3], dim=1)
            loss = F.mse_loss(w_pred, label_batch)
            loss.backward()
            optimizer.step()
            total += loss.item()
        avg = total / len(loader)
        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), args.out)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'  epoch {epoch+1:3d}/{args.epochs}  loss={avg:.6f}  best={best_loss:.6f}')

    print(f'\n[v2] Saved best checkpoint to {args.out}')

    # Sanity check on canonical test prompts.
    model.load_state_dict(torch.load(args.out, map_location=args.device))
    model.eval()
    print('\n[v2] Sample predictions (flattened → should be less peaked):')
    for p in ['a symmetric vase', 'a smooth sphere', 'a compact stone',
              'a polished marble', 'a balanced chess piece', 'a dense bowling ball']:
        pred = model.predict(p)
        w = pred['weights']
        print(f"  {p:35s}: sym={w['symmetry']:.3f} smo={w['smoothness']:.3f} com={w['compactness']:.3f}  -> {pred['dominant_property']}")


if __name__ == '__main__':
    main()
