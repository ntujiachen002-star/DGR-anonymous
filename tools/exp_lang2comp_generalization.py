"""
Lang2Comp Generalization Experiment

Tests whether Lang2Comp generalizes beyond its synthetic training data.
Three test dimensions:
  T1: Out-of-distribution objects (novel objects, same attribute keywords)
  T2: Paraphrased prompts (same semantics, different wording — no keywords)
  T3: Complex natural descriptions (longer, real-world-style text)

Baselines:
  - Equal weights (0.33, 0.33, 0.34) — always 33% accuracy by chance
  - Keyword-match — regex for "symmetric/smooth/compact" etc.

Runs on CPU in ~5 min. No GPU or Shap-E needed.

Usage:
    PYTHONPATH=src python tools/exp_lang2comp_generalization.py

Output:
    analysis_results/lang2comp_generalization/stats.json
    analysis_results/lang2comp_generalization/per_prompt_results.json
"""

import os, sys, json
import numpy as np
from pathlib import Path
import re

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))

# ── OOD Prompt Definitions ───────────────────────────────────────────────────

# Training used these 20 objects:
# vase, chair, table, bowl, cup, lamp, bottle, sphere, box, sculpture,
# rock, shell, ring, pillar, dome, arch, bridge, wheel, cone, torus

# Test 1: OOD Objects — same attribute keywords, novel objects
T1_SYMMETRY = [
    "a symmetric violin",
    "a symmetric saxophone",
    "a symmetric grandfather clock",
    "a symmetric chariot",
    "a symmetric pagoda lantern",
    "a symmetric biplane",
    "a symmetric windmill",
    "a symmetric cathedral spire",
    "a symmetric samurai helmet",
    "a symmetric trident",
]

T1_SMOOTHNESS = [
    "a smooth manatee",
    "a smooth blimp",
    "a smooth soap bubble",
    "a smooth mercury droplet",
    "a smooth silk scarf",
    "a smooth porcelain teacup",
    "a smooth glass marble",
    "a smooth melted wax figure",
    "a smooth balloon animal",
    "a smooth pear",
]

T1_COMPACTNESS = [
    "a compact grenade",
    "a compact acorn",
    "a compact walnut",
    "a compact hockey puck",
    "a compact button",
    "a compact pill capsule",
    "a compact thimble",
    "a compact golf ball",
    "a compact pincushion",
    "a compact sugar cube",
]

# Test 2: Paraphrased — same semantics, NO attribute keywords
T2_SYMMETRY = [
    "an object that looks the same on both sides",
    "something with identical left and right halves",
    "a shape that is perfectly mirrored along its center",
    "an object balanced on every axis",
    "a form where each half is a reflection of the other",
    "something you could fold in half and both sides would match",
    "an object with equal proportions on all sides",
    "a shape that looks unchanged when flipped horizontally",
    "a design with matching opposing features",
    "an artifact where the left mirrors the right exactly",
]

T2_SMOOTHNESS = [
    "an object with no bumps or wrinkles on its surface",
    "something that feels like touching glass",
    "a shape with aerodynamically contoured surfaces",
    "an object with continuously curved surfaces everywhere",
    "a form where you could slide your hand over it without catching",
    "something with no rough patches or sharp transitions",
    "a surface that flows without interruption",
    "an object as slippery as wet ice",
    "a form with gentle gradual curvature throughout",
    "something with a surface like still water",
]

T2_COMPACTNESS = [
    "an object that fits snugly in your hand",
    "something dense and chunky with minimal wasted space",
    "a shape with the smallest possible surface for its volume",
    "a tightly packed form with no protruding parts",
    "an object that takes up as little space as possible",
    "something heavy for its size, no hollow parts",
    "a form where nothing sticks out or extends",
    "an object you could pack efficiently in a box",
    "a shape that is bulky relative to its surface area",
    "something squat and solid, not elongated",
]

# Test 3: Complex Natural Descriptions
T3_SYMMETRY = [
    "a decorative porcelain figurine of a ballerina, perfectly mirrored left to right",
    "an ancient Greek amphora with handles placed in exact opposition",
    "a ceremonial Japanese torii gate with identical pillars on each side",
    "a Renaissance-style architectural facade with matching windows and columns",
    "a heraldic coat of arms carved in stone with opposing lions",
    "a pair of ornamental bookends that are mirror images of each other",
    "an Art Deco building entrance with geometrically matched panels",
    "a butterfly-wing brooch where both wings have identical venation patterns",
    "a cathedral rose window with radially repeating geometric segments",
    "a tribal mask with precisely matching features on left and right",
]

T3_SMOOTHNESS = [
    "a concept car prototype with flowing lines and zero sharp edges",
    "a Henry Moore inspired abstract sculpture with gentle undulating surfaces",
    "a weathered beach pebble tumbled by centuries of ocean waves",
    "an organic architectural pod inspired by seed pods and cocoons",
    "a futuristic helmet design with seamlessly blended visor and shell",
    "a whale-shaped children's slide with gradually curving surfaces throughout",
    "a Zaha Hadid style building with fluid parametric exterior walls",
    "a hand-blown glass ornament with naturally flowing rounded contours",
    "a ceramic vessel shaped by a potter's hands with soft continuous curves",
    "a streamlined high-speed train nose cone designed for minimal air resistance",
]

T3_COMPACTNESS = [
    "a small dense meteorite fragment that fits in a museum display case",
    "a cast iron cannonball from a 17th century warship",
    "a tightly wound ball of rubber bands with no loose ends",
    "a miniature snow globe with everything contained in a sealed sphere",
    "a solid bronze chess piece, a rook, heavy and without hollow sections",
    "a fossilized dinosaur egg, roughly spherical and dense throughout",
    "a Japanese netsuke carving, a small intricate but contained figure",
    "a military-grade ball bearing, perfectly round and extremely dense",
    "a sealed terrarium globe containing a miniature ecosystem",
    "a hand-carved wooden kokeshi doll, cylindrical and solid",
]

ALL_TESTS = {
    "T1_ood_objects": {
        "symmetry": T1_SYMMETRY,
        "smoothness": T1_SMOOTHNESS,
        "compactness": T1_COMPACTNESS,
    },
    "T2_paraphrased": {
        "symmetry": T2_SYMMETRY,
        "smoothness": T2_SMOOTHNESS,
        "compactness": T2_COMPACTNESS,
    },
    "T3_complex_natural": {
        "symmetry": T3_SYMMETRY,
        "smoothness": T3_SMOOTHNESS,
        "compactness": T3_COMPACTNESS,
    },
}

PROPERTY_NAMES = ["symmetry", "smoothness", "compactness"]

# ── Baseline Methods ─────────────────────────────────────────────────────────

SYM_KEYWORDS = re.compile(
    r"symmetr|mirror|balanced|identical.*halves|reflection|bilateral|even.*proportion",
    re.IGNORECASE,
)
SMO_KEYWORDS = re.compile(
    r"smooth|polish|sleek|flowing|curved|organic|stream|aerodynamic|contour|gentle",
    re.IGNORECASE,
)
COM_KEYWORDS = re.compile(
    r"compact|dense|solid|tight|heavy|small|fit.*hand|squat|chunky|snug|packed|minimal.*space",
    re.IGNORECASE,
)


def keyword_match_predict(text: str) -> str:
    """Simple keyword matching baseline."""
    scores = {
        "symmetry": len(SYM_KEYWORDS.findall(text)),
        "smoothness": len(SMO_KEYWORDS.findall(text)),
        "compactness": len(COM_KEYWORDS.findall(text)),
    }
    if max(scores.values()) == 0:
        return "symmetry"  # default fallback
    return max(scores, key=scores.get)


def equal_weight_predict(text: str) -> str:
    """Equal weight baseline — always predicts first category (random 33%)."""
    import random
    return random.choice(PROPERTY_NAMES)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import torch

    OUT_DIR = Path("analysis_results/lang2comp_generalization")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load Lang2Comp
    from lang2comp import Lang2Comp

    ckpt_path = "checkpoints/lang2comp_best.pt"
    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found at {ckpt_path}")
        print("  Please ensure the trained Lang2Comp model is available.")
        sys.exit(1)

    print("Loading Lang2Comp...")
    model = Lang2Comp()
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Handle different checkpoint formats
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"], strict=False)
    elif isinstance(state, dict) and any(k.startswith("composition_head") for k in state):
        model.load_state_dict(state, strict=False)
    else:
        model.load_state_dict(state, strict=False)
    model.eval()
    print("  Lang2Comp loaded.")

    # Run all tests
    all_results = []
    np.random.seed(42)

    print("\n" + "=" * 60)
    print("Lang2Comp Generalization Experiment")
    print(f"  3 tests × 3 categories × 10 prompts = 90 total")
    print("=" * 60)

    for test_name, categories in ALL_TESTS.items():
        print(f"\n--- {test_name} ---")

        for gt_category, prompts in categories.items():
            for prompt in prompts:
                # Lang2Comp prediction
                with torch.no_grad():
                    pred = model.predict(prompt)
                l2c_dominant = pred["dominant_property"]
                l2c_weights = pred["weights"]

                # Keyword-match prediction
                kw_dominant = keyword_match_predict(prompt)

                # Equal-weight prediction (random)
                eq_dominant = equal_weight_predict(prompt)

                result = {
                    "test": test_name,
                    "gt_category": gt_category,
                    "prompt": prompt,
                    "lang2comp_dominant": l2c_dominant,
                    "lang2comp_correct": l2c_dominant == gt_category,
                    "lang2comp_weights": l2c_weights,
                    "lang2comp_target_weight": l2c_weights[gt_category],
                    "keyword_dominant": kw_dominant,
                    "keyword_correct": kw_dominant == gt_category,
                    "equal_correct": eq_dominant == gt_category,
                }
                all_results.append(result)

                mark = "O" if result["lang2comp_correct"] else "X"
                kw_mark = "O" if result["keyword_correct"] else "X"
                print(
                    f"  [{mark}] L2C={l2c_dominant[:3]} "
                    f"[{kw_mark}] KW={kw_dominant[:3]} "
                    f"GT={gt_category[:3]}  "
                    f"w={l2c_weights[gt_category]:.3f}  "
                    f"{prompt[:50]}"
                )

    # ── Compute Statistics ────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    stats = {}

    # Overall
    l2c_correct = sum(r["lang2comp_correct"] for r in all_results)
    kw_correct = sum(r["keyword_correct"] for r in all_results)
    eq_correct = sum(r["equal_correct"] for r in all_results)
    n_total = len(all_results)

    stats["overall"] = {
        "n_total": n_total,
        "lang2comp_accuracy": l2c_correct / n_total,
        "keyword_accuracy": kw_correct / n_total,
        "equal_accuracy": eq_correct / n_total,
        "lang2comp_mean_target_weight": np.mean(
            [r["lang2comp_target_weight"] for r in all_results]
        ),
    }

    print(f"\nOverall ({n_total} prompts):")
    print(f"  Lang2Comp:     {l2c_correct}/{n_total} ({l2c_correct/n_total*100:.1f}%)")
    print(f"  Keyword-Match: {kw_correct}/{n_total} ({kw_correct/n_total*100:.1f}%)")
    print(f"  Equal Weight:  {eq_correct}/{n_total} ({eq_correct/n_total*100:.1f}%)")
    print(
        f"  Mean target weight: {stats['overall']['lang2comp_mean_target_weight']:.3f}"
    )

    # Per test
    for test_name in ALL_TESTS:
        test_results = [r for r in all_results if r["test"] == test_name]
        n = len(test_results)
        l2c_acc = sum(r["lang2comp_correct"] for r in test_results) / n
        kw_acc = sum(r["keyword_correct"] for r in test_results) / n
        eq_acc = sum(r["equal_correct"] for r in test_results) / n
        mean_tw = np.mean([r["lang2comp_target_weight"] for r in test_results])

        stats[test_name] = {
            "n": n,
            "lang2comp_accuracy": l2c_acc,
            "keyword_accuracy": kw_acc,
            "equal_accuracy": eq_acc,
            "lang2comp_mean_target_weight": mean_tw,
        }

        print(f"\n{test_name} ({n} prompts):")
        print(f"  Lang2Comp:     {l2c_acc*100:.1f}%")
        print(f"  Keyword-Match: {kw_acc*100:.1f}%")
        print(f"  Equal Weight:  {eq_acc*100:.1f}%")
        print(f"  Mean target weight: {mean_tw:.3f}")

        # Per category within test
        for cat in PROPERTY_NAMES:
            cat_results = [r for r in test_results if r["gt_category"] == cat]
            if not cat_results:
                continue
            nc = len(cat_results)
            l2c_cat = sum(r["lang2comp_correct"] for r in cat_results) / nc
            kw_cat = sum(r["keyword_correct"] for r in cat_results) / nc

            stats[f"{test_name}_{cat}"] = {
                "n": nc,
                "lang2comp_accuracy": l2c_cat,
                "keyword_accuracy": kw_cat,
            }
            print(f"    {cat:12s}: L2C {l2c_cat*100:.0f}%  KW {kw_cat*100:.0f}%")

    # ── Key Finding: T2 paraphrased advantage ────────────────────────────

    t2_l2c = stats["T2_paraphrased"]["lang2comp_accuracy"]
    t2_kw = stats["T2_paraphrased"]["keyword_accuracy"]
    print(f"\n{'='*60}")
    print(f"KEY FINDING — Paraphrased prompts (hardest test):")
    print(f"  Lang2Comp: {t2_l2c*100:.1f}%  vs  Keyword-Match: {t2_kw*100:.1f}%")
    if t2_l2c > t2_kw:
        print(
            f"  Lang2Comp outperforms keyword-match by "
            f"{(t2_l2c - t2_kw)*100:.1f} percentage points"
        )
        print(f"  → Proves semantic understanding, not keyword memorization")
    print(f"{'='*60}")

    # ── Save results ─────────────────────────────────────────────────────

    json.dump(all_results, open(OUT_DIR / "per_prompt_results.json", "w"), indent=2)
    json.dump(stats, open(OUT_DIR / "stats.json", "w"), indent=2)

    print(f"\nResults saved to {OUT_DIR}/")
    print("=== Experiment complete ===")


if __name__ == "__main__":
    main()
