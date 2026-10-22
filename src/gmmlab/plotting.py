"""Small plotting helpers shared across experiment scripts.

Extracted from experiments/stage4_sanity_check.py once experiments/
rq1_cluster_overlap.py needed the identical ellipse-drawing code. Two copies
of the same function is exactly the kind of duplication the project's own
habit ("small pieces, reused, not duplicated") argues against.
"""

import numpy as np
from matplotlib.patches import Ellipse


def covariance_ellipse(ax, mean, cov, n_std=2.0, **kwargs):
    """Draw one Gaussian's spread as an ellipse on `ax`.

    This is lessons/01_latent_variables_and_gmm.md, Section 1.2's geometry, drawn:
    eigenvectors of `cov` are the ellipse's axis directions, sqrt(eigenvalues)
    are the axis half-lengths. `n_std` just sets how many of those to draw,
    a visual choice for a sanity-check plot, not a precise confidence region.
    """
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending order
    # largest eigenvalue's eigenvector sets the ellipse's rotation angle
    angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
    width, height = 2 * n_std * np.sqrt(eigvals[::-1])  # diameters, largest axis first
    ax.add_patch(
        Ellipse(xy=mean, width=width, height=height, angle=angle, fill=False, **kwargs)
    )
