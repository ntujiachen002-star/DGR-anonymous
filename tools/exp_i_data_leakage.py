"""
Experiment I: Lang2Comp Training/Test Overlap Detection
Checks whether evaluation prompts are semantically similar to training templates.
CPU-only: uses sentence-transformers embeddings.
"""
import os, sys, json, numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "analysis_results" / "data_leakage"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Import project modules
    from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS
    from lang2comp import TRAINING_TEMPLATES, OBJECTS, generate_training_texts

    eval_prompts = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
    print(f"Evaluation prompts: {len(eval_prompts)}")

    # Generate training texts (same function used during actual training)
    np.random.seed(42)
    import random
    random.seed(42)
    train_samples = generate_training_texts(10000)
    train_texts = list(set(s['text'] for s in train_samples))  # deduplicate
    print(f"Unique training texts: {len(train_texts)}")

    # Also collect raw templates (before object substitution)
    raw_templates = []
    for category, templates in TRAINING_TEMPLATES.items():
        for t in templates:
            for obj in OBJECTS:
                raw_templates.append(t.format(object=obj))
    raw_templates = list(set(raw_templates))
    print(f"Raw template expansions: {len(raw_templates)}")

    # Encode with sentence-transformers
    print("Loading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding evaluation prompts...")
    eval_embeds = model.encode(eval_prompts, normalize_embeddings=True, show_progress_bar=True)

    print("Encoding training texts...")
    train_embeds = model.encode(train_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    print("Encoding raw templates...")
    template_embeds = model.encode(raw_templates, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    # Compute cosine similarity (embeddings are already normalized)
    print("Computing similarities...")
    sims_train = eval_embeds @ train_embeds.T  # (n_eval, n_train)
    sims_template = eval_embeds @ template_embeds.T  # (n_eval, n_template)

    max_sims_train = sims_train.max(axis=1)
    max_sims_template = sims_template.max(axis=1)

    # Check exact string matches
    eval_set = set(p.lower().strip() for p in eval_prompts)
    train_set = set(t.lower().strip() for t in train_texts)
    exact_matches = eval_set & train_set
    print(f"\nExact string matches: {len(exact_matches)}")
    if exact_matches:
        for m in sorted(exact_matches):
            print(f"  EXACT: '{m}'")

    # Summary statistics
    report = {
        "n_eval_prompts": len(eval_prompts),
        "n_unique_train_texts": len(train_texts),
        "n_raw_templates": len(raw_templates),
        "n_exact_matches": len(exact_matches),
        "exact_matches": sorted(exact_matches),

        # vs generated training texts
        "train_max_sim_mean": float(max_sims_train.mean()),
        "train_max_sim_median": float(np.median(max_sims_train)),
        "train_max_sim_std": float(max_sims_train.std()),
        "train_n_above_095": int((max_sims_train > 0.95).sum()),
        "train_n_above_090": int((max_sims_train > 0.90).sum()),
        "train_n_above_085": int((max_sims_train > 0.85).sum()),
        "train_n_above_080": int((max_sims_train > 0.80).sum()),

        # vs raw templates
        "template_max_sim_mean": float(max_sims_template.mean()),
        "template_max_sim_median": float(np.median(max_sims_template)),
        "template_max_sim_std": float(max_sims_template.std()),
        "template_n_above_095": int((max_sims_template > 0.95).sum()),
        "template_n_above_090": int((max_sims_template > 0.90).sum()),
        "template_n_above_085": int((max_sims_template > 0.85).sum()),

        # Per-prompt details (top 20 most similar)
        "per_prompt_top20": [],
    }

    # Top 20 most similar eval prompts
    top_indices = np.argsort(max_sims_train)[::-1][:20]
    for idx in top_indices:
        top5_train_idx = np.argsort(sims_train[idx])[::-1][:5]
        report["per_prompt_top20"].append({
            "eval_prompt": eval_prompts[idx],
            "max_sim_train": float(max_sims_train[idx]),
            "max_sim_template": float(max_sims_template[idx]),
            "top5_train_matches": [
                {"text": train_texts[j], "sim": float(sims_train[idx, j])}
                for j in top5_train_idx
            ]
        })

    # Save
    with open(OUT_DIR / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n=== DATA LEAKAGE REPORT ===")
    print(f"Exact string matches:     {report['n_exact_matches']}")
    print(f"\nvs Training Texts:")
    print(f"  Mean max similarity:    {report['train_max_sim_mean']:.4f} ± {report['train_max_sim_std']:.4f}")
    print(f"  Median max similarity:  {report['train_max_sim_median']:.4f}")
    print(f"  Prompts with sim > 0.95: {report['train_n_above_095']}")
    print(f"  Prompts with sim > 0.90: {report['train_n_above_090']}")
    print(f"  Prompts with sim > 0.85: {report['train_n_above_085']}")
    print(f"  Prompts with sim > 0.80: {report['train_n_above_080']}")
    print(f"\nvs Raw Templates:")
    print(f"  Mean max similarity:    {report['template_max_sim_mean']:.4f} ± {report['template_max_sim_std']:.4f}")
    print(f"  Prompts with sim > 0.95: {report['template_n_above_095']}")
    print(f"  Prompts with sim > 0.90: {report['template_n_above_090']}")
    print(f"  Prompts with sim > 0.85: {report['template_n_above_085']}")

    print(f"\nTop 5 most similar eval prompts:")
    for item in report["per_prompt_top20"][:5]:
        print(f"  '{item['eval_prompt']}' → sim={item['max_sim_train']:.4f}")
        print(f"    closest: '{item['top5_train_matches'][0]['text']}' (sim={item['top5_train_matches'][0]['sim']:.4f})")

    # Interpretation
    mean_sim = report['train_max_sim_mean']
    if mean_sim < 0.80:
        verdict = "NO significant overlap detected. Evaluation prompts are semantically distinct from training templates."
    elif mean_sim < 0.90:
        verdict = "MILD overlap detected. Some evaluation prompts share domain vocabulary with training templates, but are structurally different."
    else:
        verdict = "HIGH overlap detected. Consider re-evaluating on held-out prompts."

    print(f"\nVERDICT: {verdict}")
    report["verdict"] = verdict

    with open(OUT_DIR / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
