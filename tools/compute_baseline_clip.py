#!/usr/bin/env python
"""M2: Compute CLIP scores for baseline meshes."""
import os, sys, json, glob
import numpy as np
import torch
import clip
from PIL import Image
import trimesh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def render_mesh_matplotlib(mesh_path, angle_deg=0, resolution=224):
    mesh = trimesh.load(mesh_path)
    verts = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    center = verts.mean(axis=0)
    verts = verts - center
    scale = np.abs(verts).max()
    if scale > 0:
        verts = verts / scale
    if len(faces) > 5000:
        idx = np.random.choice(len(faces), 5000, replace=False)
        faces = faces[idx]

    fig = plt.figure(figsize=(resolution/72, resolution/72), dpi=72)
    ax = fig.add_subplot(111, projection='3d')
    polys = verts[faces]
    pc = Poly3DCollection(polys, alpha=0.8, facecolor='lightblue', edgecolor='gray', linewidth=0.1)
    ax.add_collection3d(pc)
    ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
    ax.view_init(elev=20, azim=angle_deg)
    ax.axis('off')
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=72)
    plt.close(fig)
    buf.seek(0)
    return buf


def compute_clip_score(mesh_path, prompt, model, preprocess, device):
    scores = []
    for angle in [0, 90, 180, 270]:
        try:
            buf = render_mesh_matplotlib(mesh_path, angle)
            image = Image.open(buf).convert('RGB')
            image_input = preprocess(image).unsqueeze(0).to(device)
            text_input = clip.tokenize([prompt]).to(device)
            with torch.no_grad():
                img_feat = model.encode_image(image_input)
                txt_feat = model.encode_text(text_input)
                img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)
                txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                sim = (img_feat @ txt_feat.T).item()
                scores.append(sim)
        except Exception as e:
            print(f"  Render failed for angle {angle}: {e}")
    return float(np.mean(scores)) if scores else 0.0


def main():
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Loading CLIP...")
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Load baseline metrics
    bl_path = 'results/baseline_all_metrics.json'
    if not os.path.exists(bl_path):
        bl_path = 'results/full/baseline_all_metrics.json'
    with open(bl_path) as f:
        bl_data = json.load(f)

    print(f"Processing {len(bl_data)} baseline entries...")
    updated = 0
    for i, entry in enumerate(bl_data):
        if entry.get('clip_score', 0) > 0:
            continue  # already has clip score

        prompt = entry['prompt']
        seed = entry['seed']
        method = entry.get('method', 'baseline')

        # Find the mesh file
        from prompts_gpteval3d import SYMMETRY_PROMPTS, SMOOTHNESS_PROMPTS, COMPACTNESS_PROMPTS
        if prompt in SYMMETRY_PROMPTS:
            cat = 'symmetry'
        elif prompt in SMOOTHNESS_PROMPTS:
            cat = 'smoothness'
        elif prompt in COMPACTNESS_PROMPTS:
            cat = 'compactness'
        else:
            cat = 'unknown'

        mesh_path = f'results/full/baseline/{cat}/baseline_seed{seed}.obj'
        if not os.path.exists(mesh_path) or os.path.getsize(mesh_path) < 200:
            print(f"  [{i+1}] Mesh not found: {mesh_path}")
            continue

        score = compute_clip_score(mesh_path, prompt, model, preprocess, device)
        entry['clip_score'] = score
        updated += 1
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(bl_data)}] {prompt[:40]}... clip={score:.4f}")

    # Save updated metrics
    with open(bl_path, 'w') as f:
        json.dump(bl_data, f, indent=2)

    # Print summary
    clips = [e['clip_score'] for e in bl_data if e.get('clip_score', 0) > 0]
    if clips:
        print(f"\nBaseline CLIP: mean={np.mean(clips):.4f}, std={np.std(clips):.4f}, n={len(clips)}")
    print(f"Updated {updated} entries. Saved to {bl_path}")


if __name__ == '__main__':
    main()
