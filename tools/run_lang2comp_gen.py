import torch, json
from lang2comp import Lang2Comp
import os

model = Lang2Comp()
model.load_state_dict(torch.load("checkpoints/lang2comp_retrained.pt", map_location="cpu"))
model.eval()

t1 = ["a symmetric airplane", "a smooth guitar", "a compact backpack",
      "a balanced lighthouse", "a polished telescope", "a dense anchor"]
t2 = ["make this vase more symmetrical", "I want smoother curves on the helmet",
      "reduce the surface area of this blob", "this chair needs to be more balanced",
      "refine the roundness of this ball", "tighten up this loose shape"]
t3 = ["a chair", "something cool", "make it better",
      "a detailed dragon", "abstract art", "hello world"]

results = {}
for test_name, prompts in [("T1_seen_template", t1), ("T2_free_form", t2), ("T3_adversarial", t3)]:
    preds = []
    for p in prompts:
        pred = model.predict(p)
        preds.append({"prompt": p, "weights": pred["weights"], "dominant_property": pred["dominant_property"]})
        w = pred["weights"]
        print(f"  [{test_name}] {p:45s}: sym={w['symmetry']:.2f} smo={w['smoothness']:.2f} com={w['compactness']:.2f} -> {pred['dominant_property']}")
    results[test_name] = preds

os.makedirs("analysis_results/lang2comp_rerun", exist_ok=True)
with open("analysis_results/lang2comp_rerun/generalization.json", "w") as f:
    json.dump(results, f, indent=2)
print("Saved")
