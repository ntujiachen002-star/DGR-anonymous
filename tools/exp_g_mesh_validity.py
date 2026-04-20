"""
Experiment G: Mesh Topology Validity Check
Verifies that vertex optimization preserves mesh integrity (no degenerate faces, consistent winding).
CPU-only: analyzes existing .obj files.
"""
import os, sys, json, numpy as np, trimesh
from pathlib import Path
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_FULL = PROJECT_ROOT / "results" / "full"
OUT_DIR = PROJECT_ROOT / "analysis_results" / "mesh_validity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

METHODS = ["baseline", "diffgeoreward", "handcrafted", "sym_only", "smooth_only", "compact_only"]
CATEGORIES = ["symmetry", "smoothness", "compactness"]


def check_mesh(obj_path, baseline_path=None):
    """Compute mesh validity metrics for a single .obj file."""
    mesh = trimesh.load(str(obj_path), force='mesh')
    tri_areas = trimesh.triangles.area(mesh.triangles)

    report = {
        "path": str(obj_path),
        "n_vertices": len(mesh.vertices),
        "n_faces": len(mesh.faces),
        # 1. Degenerate faces (area < 1e-8)
        "n_degenerate_faces": int((tri_areas < 1e-8).sum()),
        "degenerate_face_ratio": float((tri_areas < 1e-8).mean()),
        # 2. Watertight (closed surface)
        "is_watertight": bool(mesh.is_watertight),
        # 3. Winding consistency
        "is_winding_consistent": bool(mesh.is_winding_consistent),
        # 4. Volume (negative = inverted normals)
        "volume": float(mesh.volume) if mesh.is_watertight else None,
        # 5. Euler characteristic
        "euler_number": int(mesh.euler_number),
        # 6. Min/max face area
        "min_face_area": float(tri_areas.min()),
        "max_face_area": float(tri_areas.max()),
        "mean_face_area": float(tri_areas.mean()),
    }

    # 7. Hausdorff distance to baseline (shape deformation measure)
    if baseline_path and Path(baseline_path).exists():
        try:
            bl = trimesh.load(str(baseline_path), force='mesh')
            pts_a = mesh.sample(2000)
            pts_b = bl.sample(2000)
            from scipy.spatial import cKDTree
            d_ab = cKDTree(pts_b).query(pts_a)[0].max()
            d_ba = cKDTree(pts_a).query(pts_b)[0].max()
            report["hausdorff_to_baseline"] = float(max(d_ab, d_ba))
            if mesh.is_watertight and bl.is_watertight:
                report["volume_change_ratio"] = float(
                    (mesh.volume - bl.volume) / max(abs(bl.volume), 1e-8)
                )
            else:
                report["volume_change_ratio"] = None
        except Exception as e:
            report["hausdorff_to_baseline"] = None
            report["volume_change_ratio"] = None
            report["hausdorff_error"] = str(e)

    return report


def main():
    all_reports = []

    for method in METHODS:
        method_dir = RESULTS_FULL / method
        if not method_dir.exists():
            print(f"  [SKIP] {method_dir} not found")
            continue

        for cat in CATEGORIES:
            cat_dir = method_dir / cat
            if not cat_dir.exists():
                continue

            for obj_path in sorted(cat_dir.glob("*.obj")):
                # Find corresponding baseline by seed
                baseline_path = None
                if method != "baseline":
                    # Extract seed from filename (e.g., diffgeoreward_seed42.obj -> seed42)
                    import re
                    seed_match = re.search(r'seed(\d+)', obj_path.name)
                    if seed_match:
                        seed = seed_match.group(1)
                        baseline_path = RESULTS_FULL / "baseline" / cat / f"baseline_seed{seed}.obj"

                report = check_mesh(obj_path, baseline_path)
                report["method"] = method
                report["category"] = cat
                report["filename"] = obj_path.name
                all_reports.append(report)
                status = "WT" if report["is_watertight"] else "OPEN"
                degen = report["n_degenerate_faces"]
                print(f"  [{method:15s}] {cat}/{obj_path.name}: {status}, degen={degen}, V={report['n_vertices']}, F={report['n_faces']}")

    # Save all reports
    with open(OUT_DIR / "all_reports.json", 'w') as f:
        json.dump(all_reports, f, indent=2)

    # Summary table
    print("\n=== SUMMARY ===")
    print(f"{'Method':<18} {'Count':>6} {'Watertight':>11} {'Winding OK':>11} {'Degen Ratio':>12} {'Mean Hausdorff':>15}")

    for method in METHODS:
        rows = [r for r in all_reports if r["method"] == method]
        if not rows:
            continue
        n = len(rows)
        n_wt = sum(1 for r in rows if r["is_watertight"])
        n_wc = sum(1 for r in rows if r["is_winding_consistent"])
        degen_ratios = [r["degenerate_face_ratio"] for r in rows]
        hausdorff = [r.get("hausdorff_to_baseline") for r in rows if r.get("hausdorff_to_baseline") is not None]

        h_str = f"{np.mean(hausdorff):.4f}±{np.std(hausdorff):.4f}" if hausdorff else "—"
        print(f"{method:<18} {n:>6} {n_wt:>5}/{n} ({100*n_wt/n:.0f}%) {n_wc:>5}/{n} ({100*n_wc/n:.0f}%) {np.mean(degen_ratios):>11.6f} {h_str:>15}")

    # Save summary CSV
    summary_rows = []
    for method in METHODS:
        rows = [r for r in all_reports if r["method"] == method]
        if not rows:
            continue
        n = len(rows)
        n_wt = sum(1 for r in rows if r["is_watertight"])
        n_wc = sum(1 for r in rows if r["is_winding_consistent"])
        degen = [r["degenerate_face_ratio"] for r in rows]
        hausdorff = [r.get("hausdorff_to_baseline") for r in rows if r.get("hausdorff_to_baseline") is not None]
        vol_change = [r.get("volume_change_ratio") for r in rows if r.get("volume_change_ratio") is not None]

        summary_rows.append({
            "method": method,
            "n": n,
            "watertight_pct": 100 * n_wt / n,
            "winding_consistent_pct": 100 * n_wc / n,
            "degenerate_face_ratio_mean": float(np.mean(degen)),
            "degenerate_face_ratio_std": float(np.std(degen)),
            "hausdorff_mean": float(np.mean(hausdorff)) if hausdorff else None,
            "hausdorff_std": float(np.std(hausdorff)) if hausdorff else None,
            "volume_change_mean": float(np.mean(vol_change)) if vol_change else None,
            "volume_change_std": float(np.std(vol_change)) if vol_change else None,
        })

    with open(OUT_DIR / "summary.json", 'w') as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nResults saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
