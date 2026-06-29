"""
plot_utils.py
-------------
Shared figure utilities for fUS analysis scripts.
"""

from pathlib import Path

import matplotlib.pyplot as plt


def savefig(fig: plt.Figure, stem: Path, dpi: int = 180) -> None:
    """Save a figure as both PNG and PDF."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix('.png'), dpi=dpi, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight')