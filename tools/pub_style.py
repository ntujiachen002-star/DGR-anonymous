"""
Publication-quality figure style for NeurIPS 2026 submission.
References: SciencePlots, DreamDPO (ICML'25), MeshFormer (NeurIPS'24)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl

# ── SciencePlots 6-color palette (colorblind-friendly) ───────────────────────
PALETTE = [
    '#0C5DA5',  # blue
    '#00B945',  # green
    '#FF9500',  # orange
    '#FF2C00',  # red
    '#845B97',  # purple
    '#474747',  # dark gray
    '#9e9e9e',  # light gray
]

# Method-specific colors
METHOD_COLORS = {
    'baseline':       '#9e9e9e',
    'diffgeoreward':  '#0C5DA5',
    'handcrafted':    '#00B945',
    'sym_only':       '#FF2C00',
    'smooth_only':    '#FF9500',
    'compact_only':   '#845B97',
    'laplacian':      '#474747',
}

METHOD_LABELS = {
    'baseline':       'Baseline',
    'diffgeoreward':  'DGR (Ours)',
    'handcrafted':    'Equal Weights',
    'sym_only':       'Sym-Only',
    'smooth_only':    'Smo-Only',
    'compact_only':   'Com-Only',
}

METRIC_COLORS = {
    'symmetry':    '#FF2C00',
    'smoothness':  '#0C5DA5',
    'compactness': '#00B945',
}


def setup_style():
    """Apply NeurIPS publication style."""
    use_latex = False
    try:
        # Test if LaTeX is available
        import shutil
        if shutil.which('latex'):
            use_latex = True
    except Exception:
        pass

    params = {
        # Font — use LaTeX rendering if available
        'text.usetex': use_latex,
        'font.family': 'serif',
        'font.serif': ['Computer Modern Roman', 'Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'cm',

        # Font sizes (match NeurIPS 10pt body)
        'font.size': 9,
        'axes.labelsize': 9,
        'axes.titlesize': 9,
        'legend.fontsize': 7.5,
        'xtick.labelsize': 7.5,
        'ytick.labelsize': 7.5,

        # Lines
        'lines.linewidth': 1.3,
        'lines.markersize': 4,

        # Axes
        'axes.linewidth': 0.5,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#1a1a1a',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.prop_cycle': mpl.cycler(color=PALETTE),

        # Ticks — inward, with minor ticks
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'xtick.minor.size': 2,
        'ytick.major.size': 3.5,
        'ytick.minor.size': 2,
        'xtick.major.width': 0.5,
        'xtick.minor.width': 0.35,
        'ytick.major.width': 0.5,
        'ytick.minor.width': 0.35,
        'xtick.color': '#333333',
        'ytick.color': '#333333',

        # Grid — very subtle
        'axes.grid': False,
        'grid.alpha': 0.25,
        'grid.linewidth': 0.3,
        'grid.color': '#CCCCCC',

        # Legend
        'legend.frameon': False,
        'legend.handlelength': 1.5,
        'legend.handletextpad': 0.4,
        'legend.columnspacing': 1.0,

        # Figure
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.02,
        'figure.constrained_layout.use': False,
    }
    plt.rcParams.update(params)


# ── NeurIPS sizing utilities ──────────────────────────────────────────────────
TEXTWIDTH_PT = 397.48   # NeurIPS \textwidth
COLWIDTH_PT  = 397.48   # single-column

def figsize(fraction=1.0, aspect=0.618, subplots=(1, 1)):
    """
    Compute figure size matching NeurIPS column width.
    fraction: fraction of textwidth
    aspect: height/width ratio (default: golden ratio)
    subplots: (nrows, ncols)
    """
    w = TEXTWIDTH_PT / 72.27 * fraction
    h = w * aspect * (subplots[0] / subplots[1])
    return (w, h)
