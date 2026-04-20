"""
Language-to-Composition Network

Maps text descriptions to geometric reward composition weights.
f(text) → (w1, w2, w3) ∈ Δ² (probability simplex)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class Lang2Comp(nn.Module):
    """Language-conditioned geometric reward composition.

    Takes a text description and outputs:
    - weights: (3,) softmax weights for [symmetry, smoothness, compactness]
    - params: (3,) sigmoid parameters for each property
    """

    PROPERTY_NAMES = ['symmetry', 'smoothness', 'compactness']

    def __init__(
        self,
        text_encoder_name: str = 'sentence-transformers/all-MiniLM-L6-v2',
        hidden_dim: int = 256,
        freeze_encoder: bool = True,
    ):
        super().__init__()

        self.text_encoder = SentenceTransformer(text_encoder_name, device='cpu')
        self.embed_dim = self.text_encoder.get_sentence_embedding_dimension()

        if freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        self.composition_head = nn.Sequential(
            nn.Linear(self.embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 6),  # 3 weights + 3 params
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.composition_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def encode_text(self, texts: list[str]) -> torch.Tensor:
        """Encode text to embeddings.

        Args:
            texts: list of N text strings
        Returns:
            (N, embed_dim) tensor on same device as model
        """
        with torch.no_grad():
            embeddings = self.text_encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
        # Clone to detach from inference mode so downstream autograd works
        device = next(self.composition_head.parameters()).device
        return embeddings.clone().detach().to(device)

    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Map text to composition weights and parameters.

        Args:
            texts: list of N text descriptions
        Returns:
            weights: (N, 3) softmax weights (sum to 1)
            params: (N, 3) sigmoid parameters in [0, 1]
        """
        embeddings = self.encode_text(texts)
        out = self.composition_head(embeddings)  # (N, 6)

        weights = F.softmax(out[:, :3], dim=-1)  # (N, 3)
        params = torch.sigmoid(out[:, 3:])  # (N, 3)

        return weights, params

    def predict(self, text: str) -> dict:
        """Single text prediction with named outputs.

        Args:
            text: single text description
        Returns:
            dict with property_name → weight mappings
        """
        weights, params = self.forward([text])
        w = weights[0].detach().cpu().numpy()
        p = params[0].detach().cpu().numpy()

        return {
            'weights': {name: float(w[i]) for i, name in enumerate(self.PROPERTY_NAMES)},
            'params': {name: float(p[i]) for i, name in enumerate(self.PROPERTY_NAMES)},
            'dominant_property': self.PROPERTY_NAMES[w.argmax()],
        }


# ============================================================
# Training Data Templates
# ============================================================

TRAINING_TEMPLATES = {
    'symmetry_high': [
        "a symmetric {object}",
        "a perfectly balanced {object}",
        "a mirror-symmetric {object}",
        "make it symmetric",
        "{object} with bilateral symmetry",
        "an evenly proportioned {object}",
    ],
    'symmetry_low': [
        "an asymmetric {object}",
        "an irregular {object}",
        "a lopsided {object}",
        "{object} with no symmetry",
    ],
    'smoothness_high': [
        "a smooth {object}",
        "a polished {object}",
        "a sleek {object}",
        "make it smoother",
        "{object} with organic curves",
        "a flowing {object}",
    ],
    'smoothness_low': [
        "a rough {object}",
        "a jagged {object}",
        "a faceted {object}",
        "{object} with sharp edges",
    ],
    'compactness_high': [
        "a compact {object}",
        "a dense {object}",
        "a solid {object}",
        "a tight {object}",
        "make it more compact",
    ],
    'compactness_low': [
        "a sprawling {object}",
        "an elongated {object}",
        "a thin {object}",
        "{object} with protruding parts",
    ],
    'composite': [
        "a smooth and symmetric {object}",
        "a compact symmetric {object}",
        "a smooth but asymmetric {object}",
        "a rough and compact {object}",
        "a smooth, symmetric, and compact {object}",
    ],
}

OBJECTS = [
    "vase", "chair", "table", "bowl", "cup", "lamp", "bottle",
    "sphere", "box", "sculpture", "rock", "shell", "ring",
    "pillar", "dome", "arch", "bridge", "wheel", "cone", "torus",
]


def generate_training_texts(n_samples: int = 10000) -> list[dict]:
    """Generate training (text, target_weights) pairs.

    Returns:
        list of dicts with 'text', 'weights' (3-tuple), 'params' (3-tuple)
    """
    import random
    samples = []

    weight_map = {
        'symmetry_high': ([0.8, 0.1, 0.1], [0.9, 0.5, 0.5]),
        'symmetry_low': ([0.8, 0.1, 0.1], [0.1, 0.5, 0.5]),
        'smoothness_high': ([0.1, 0.8, 0.1], [0.5, 0.9, 0.5]),
        'smoothness_low': ([0.1, 0.8, 0.1], [0.5, 0.1, 0.5]),
        'compactness_high': ([0.1, 0.1, 0.8], [0.5, 0.5, 0.9]),
        'compactness_low': ([0.1, 0.1, 0.8], [0.5, 0.5, 0.1]),
        'composite': ([0.33, 0.33, 0.34], [0.7, 0.7, 0.7]),
    }

    for _ in range(n_samples):
        category = random.choice(list(TRAINING_TEMPLATES.keys()))
        template = random.choice(TRAINING_TEMPLATES[category])
        obj = random.choice(OBJECTS)
        text = template.format(object=obj)

        w, p = weight_map[category]
        # Add noise for diversity
        w = [max(0.01, wi + random.gauss(0, 0.05)) for wi in w]
        p = [max(0.01, min(0.99, pi + random.gauss(0, 0.05))) for pi in p]
        w_sum = sum(w)
        w = [wi / w_sum for wi in w]

        samples.append({
            'text': text,
            'weights': w,
            'params': p,
        })

    return samples
