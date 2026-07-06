"""Python port of MATLAB's imclose (morphological closing).

Closing = dilation followed by erosion with the same structuring element.
Works on both binary (logical) and grayscale images, matching MATLAB semantics:

    J = imclose(I, SE)

Closing fills small holes and narrow gaps inside foreground regions and smooths
their outline, without changing the overall extent. It is purely a shape
operation on the image values; it does not threshold, and it does not look at
any signal statistic (mean, CV, variance, etc.). Feed it a mask (or grayscale
image) that some earlier step has already produced.
"""

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    grey_dilation,
    grey_erosion,
)


def strel_disk(radius: int) -> np.ndarray:
    """Flat disk structuring element, like strel('disk', radius).

    Returns a boolean (2*radius+1, 2*radius+1) neighborhood. Note this is a
    pure Euclidean disk; MATLAB's strel('disk', r) uses a line-segment
    approximation by default, so masks can differ by a pixel at the border.
    Pass n=0 to strel in MATLAB for the exact Euclidean disk to match this.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    L = np.arange(-radius, radius + 1)
    X, Y = np.meshgrid(L, L)
    return (X**2 + Y**2) <= radius**2


def strel_square(width: int) -> np.ndarray:
    """Square structuring element, like strel('square', width)."""
    return np.ones((width, width), dtype=bool)


def imclose(image: np.ndarray, selem: np.ndarray) -> np.ndarray:
    """Morphological closing of `image` by structuring element `selem`.

    Parameters
    ----------
    image : np.ndarray, shape (H, W)
        Binary (bool / 0-1) or grayscale image.
    selem : np.ndarray, shape (h, w)
        Structuring element neighborhood (bool / 0-1), e.g. from strel_disk.

    Returns
    -------
    np.ndarray, shape (H, W)
        Closed image, same dtype family as the input (bool in, bool out).
    """
    selem = np.asarray(selem, dtype=bool)

    is_binary = image.dtype == bool or (
        set(np.unique(image)).issubset({0, 1})
    )

    if is_binary:
        bw = image.astype(bool)
        # Dilate (pad background = False), then erode (pad foreground = True) so
        # borders are not eroded away — this matches MATLAB's border handling.
        dil = binary_dilation(bw, structure=selem, border_value=0)
        ero = binary_erosion(dil, structure=selem, border_value=1)
        return ero.astype(bool)

    # Grayscale closing.
    dil = grey_dilation(image, footprint=selem)
    ero = grey_erosion(dil, footprint=selem)
    return ero
