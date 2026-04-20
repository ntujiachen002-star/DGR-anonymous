"""
Experiment I: Lang2Comp Training/Test Overlap Detection (CPU-only, no torch needed)
Checks whether evaluation prompts are semantically similar to training templates.
"""
import os, sys, json, numpy as np, random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

OUT_DIR = PROJECT_ROOT / "analysis_results" / "data_leakage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Hardcode the prompts and templates to avoid importing torch via lang2comp
SYMMETRY_PROMPTS = [
    "a symmetric vase", "a perfectly balanced chair", "a symmetric butterfly sculpture",
    "an hourglass shape", "a symmetrical temple", "a symmetric wine glass",
    "a balanced candelabra", "a symmetric trophy", "a mirror-symmetric face mask",
    "a symmetric cathedral door", "a balanced chess piece, a king", "a symmetric crown",
    "a balanced scale", "a symmetric flower pot", "a symmetric lamp",
    "a balanced bookshelf", "a symmetric shield", "a symmetric goblet",
    "a balanced fountain", "a symmetric arch", "a symmetric chandelier",
    "a balanced table", "a symmetric pagoda", "a symmetric bell",
    "a balanced podium", "a symmetric column", "a symmetric urn",
    "a balanced picture frame", "a symmetric gazebo", "a symmetric obelisk",
    "a balanced double door", "a symmetric wedding cake", "a balanced coat hanger",
    "a symmetric window frame", "a balanced dumbbell", "a symmetric anchor",
    "a symmetric torii gate",
]

SMOOTHNESS_PROMPTS = [
    "a smooth organic blob", "a polished sphere", "a smooth river stone",
    "a smooth dolphin", "a sleek sports car body", "a smooth pebble",
    "a polished marble egg", "a smooth soap bar", "a streamlined airplane fuselage",
    "a smooth whale", "a polished gemstone", "a smooth ceramic bowl",
    "a sleek submarine", "a smooth teardrop shape", "a polished apple",
    "a smooth cloud shape", "a smooth mushroom cap", "a polished wooden spoon",
    "a smooth jellyfish bell", "a sleek yacht hull", "a smooth avocado",
    "a polished brass doorknob", "a smooth seashell", "a smooth banana",
    "a polished mirror surface", "a smooth melting chocolate drop",
    "a smooth penguin body", "a polished crystal ball", "a smooth pillow",
    "a sleek rocket nosecone", "a smooth mango", "a polished guitar body",
    "a smooth raindrop", "a smooth seal", "a polished pearl",
    "a smooth bean shape", "a smooth lava lamp blob",
]

COMPACTNESS_PROMPTS = [
    "a compact cube", "a tight ball", "a dense solid shape",
    "a minimal round object", "a solid brick", "a compact treasure chest",
    "a dense rock", "a solid wooden block", "a compact robot",
    "a dense crystal cluster", "a solid metal die", "a compact barrel",
    "a dense snowball", "a solid ice cube", "a compact backpack",
    "a dense cannonball", "a solid sphere", "a compact lantern",
    "a dense asteroid", "a solid paperweight", "a compact music box",
    "a dense bowling ball", "a solid gold nugget", "a compact toolbox",
    "a dense rubber ball", "a solid marble cube", "a compact jack-in-the-box",
    "a dense iron ingot", "a solid clay pot", "a compact birdhouse",
    "a dense geode", "a solid wooden sphere", "a compact lunch box",
    "a dense lead weight", "a solid dice", "a compact jewelry box",
]

# Training templates from lang2comp.py
TRAINING_TEMPLATES = {
    'symmetry_high': [
        "a symmetric {object}", "a perfectly balanced {object}",
        "a mirror-symmetric {object}", "make it symmetric",
        "{object} with bilateral symmetry", "an evenly proportioned {object}",
    ],
    'symmetry_low': [
        "an asymmetric {object}", "an irregular {object}",
        "a lopsided {object}", "{object} with no symmetry",
    ],
    'smoothness_high': [
        "a smooth {object}", "a polished {object}", "a sleek {object}",
        "make it smoother", "{object} with organic curves", "a flowing {object}",
    ],
    'smoothness_low': [
        "a rough {object}", "a jagged {object}",
        "a faceted {object}", "{object} with sharp edges",
    ],
    'compactness_high': [
        "a compact {object}", "a dense {object}", "a solid {object}",
        "a tight {object}", "make it more compact",
    ],
    'compactness_low': [
        "a sprawling {object}", "an elongated {object}",
        "a thin {object}", "{object} with protruding parts",
    ],
    'composite': [
        "a smooth and symmetric {object}", "a compact symmetric {object}",
        "a smooth but asymmetric {object}", "a rough and compact {object}",
        "a smooth, symmetric, and compact {object}",
    ],
}

OBJECTS = [
    "vase", "chair", "table", "bowl", "cup", "lamp", "bottle",
    "sphere", "box", "sculpture", "rock", "shell", "ring",
    "pillar", "dome", "arch", "bridge", "wheel", "cone", "torus",
]


def main():
    eval_prompts = SYMMETRY_PROMPTS + SMOOTHNESS_PROMPTS + COMPACTNESS_PROMPTS
    print(f"Evaluation prompts: {len(eval_prompts)}")

    # Generate all training template expansions
    raw_templates = []
    for category, templates in TRAINING_TEMPLATES.items():
        for t in templates:
            for obj in OBJECTS:
                raw_templates.append(t.format(object=obj))
    raw_templates = list(set(raw_templates))
    print(f"Raw template expansions: {len(raw_templates)}")

    # Generate training texts (same as training)
    random.seed(42)
    np.random.seed(42)
    train_texts_raw = []
    for _ in range(10000):
        category = random.choice(list(TRAINING_TEMPLATES.keys()))
        template = random.choice(TRAINING_TEMPLATES[category])
        obj = random.choice(OBJECTS)
        train_texts_raw.append(template.format(object=obj))
    train_texts = list(set(train_texts_raw))
    print(f"Unique training texts: {len(train_texts)}")

    # Check exact string matches first
    eval_set = set(p.lower().strip() for p in eval_prompts)
    train_set = set(t.lower().strip() for t in train_texts)
    exact_matches = eval_set & train_set
    print(f"\nExact string matches: {len(exact_matches)}")
    if exact_matches:
        for m in sorted(exact_matches):
            print(f"  EXACT: '{m}'")

    # Encode with sentence-transformers
    print("\nLoading sentence-transformers model...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Encoding evaluation prompts...")
    eval_embeds = model.encode(eval_prompts, normalize_embeddings=True, show_progress_bar=True)

    print("Encoding training texts...")
    train_embeds = model.encode(train_texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    print("Encoding raw templates...")
    template_embeds = model.encode(raw_templates, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

    # Cosine similarity
    print("Computing similarities...")
    sims_train = eval_embeds @ train_embeds.T
    sims_template = eval_embeds @ template_embeds.T

    max_sims_train = sims_train.max(axis=1)
    max_sims_template = sims_template.max(axis=1)

    report = {
        "n_eval_prompts": len(eval_prompts),
        "n_unique_train_texts": len(train_texts),
        "n_raw_templates": len(raw_templates),
        "n_exact_matches": len(exact_matches),
        "exact_matches": sorted(exact_matches),
        "train_max_sim_mean": float(max_sims_train.mean()),
        "train_max_sim_median": float(np.median(max_sims_train)),
        "train_max_sim_std": float(max_sims_train.std()),
        "train_n_above_095": int((max_sims_train > 0.95).sum()),
        "train_n_above_090": int((max_sims_train > 0.90).sum()),
        "train_n_above_085": int((max_sims_train > 0.85).sum()),
        "train_n_above_080": int((max_sims_train > 0.80).sum()),
        "template_max_sim_mean": float(max_sims_template.mean()),
        "template_max_sim_median": float(np.median(max_sims_template)),
        "template_max_sim_std": float(max_sims_template.std()),
        "template_n_above_095": int((max_sims_template > 0.95).sum()),
        "template_n_above_090": int((max_sims_template > 0.90).sum()),
        "template_n_above_085": int((max_sims_template > 0.85).sum()),
        "per_prompt_top20": [],
    }

    # Top 20 most similar
    top_indices = np.argsort(max_sims_train)[::-1][:20]
    for idx in top_indices:
        top5_idx = np.argsort(sims_train[idx])[::-1][:5]
        report["per_prompt_top20"].append({
            "eval_prompt": eval_prompts[idx],
            "max_sim_train": float(max_sims_train[idx]),
            "max_sim_template": float(max_sims_template[idx]),
            "top5_train_matches": [
                {"text": train_texts[j], "sim": float(sims_train[idx, j])}
                for j in top5_idx
            ]
        })

    with open(OUT_DIR / "report.json", 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n=== DATA LEAKAGE REPORT ===")
    print(f"Exact string matches:     {report['n_exact_matches']}")
    print(f"\nvs Training Texts:")
    print(f"  Mean max similarity:    {report['train_max_sim_mean']:.4f} +/- {report['train_max_sim_std']:.4f}")
    print(f"  Median max similarity:  {report['train_max_sim_median']:.4f}")
    print(f"  Prompts with sim > 0.95: {report['train_n_above_095']}")
    print(f"  Prompts with sim > 0.90: {report['train_n_above_090']}")
    print(f"  Prompts with sim > 0.85: {report['train_n_above_085']}")
    print(f"  Prompts with sim > 0.80: {report['train_n_above_080']}")
    print(f"\nvs Raw Templates:")
    print(f"  Mean max similarity:    {report['template_max_sim_mean']:.4f} +/- {report['template_max_sim_std']:.4f}")
    print(f"  Prompts with sim > 0.95: {report['template_n_above_095']}")
    print(f"  Prompts with sim > 0.90: {report['template_n_above_090']}")

    print(f"\nTop 5 most similar eval prompts:")
    for item in report["per_prompt_top20"][:5]:
        print(f"  '{item['eval_prompt']}' -> sim={item['max_sim_train']:.4f}")
        print(f"    closest: '{item['top5_train_matches'][0]['text']}' (sim={item['top5_train_matches'][0]['sim']:.4f})")

    mean_sim = report['train_max_sim_mean']
    if mean_sim < 0.80:
        verdict = "NO significant overlap. Evaluation prompts are semantically distinct from training."
    elif mean_sim < 0.90:
        verdict = "MILD overlap. Some domain vocabulary shared, but structurally different."
    else:
        verdict = "HIGH overlap. Consider re-evaluating on held-out prompts."

    print(f"\nVERDICT: {verdict}")
    report["verdict"] = verdict

    with open(OUT_DIR / "report.json", 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
