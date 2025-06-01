"""
Utility helpers for visualising 2‑D optimisation problems and their progress.

Functions
---------
plot_contours(func, xlim, ylim, *, levels=50, n_points=400,
              paths=None, labels=None, ax=None, title=None)
    Draws contour lines of a 2‑D objective function and (optionally) the
    optimisation paths produced by one or more algorithms.

plot_convergence(histories, *, labels=None, ax=None,
                 ylog=True, title='Objective value vs. iteration')
    Plots f(x_k) versus iteration k for several optimisation methods on a
    single set of axes so their rates of decrease can be compared.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple, Optional, List
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------#
#   Convenience converters for various input formats
# -----------------------------------------------------------------------------#
def _to_path_array(path_like) -> np.ndarray:
    """
    Convert *path_like* into a 2‑column NumPy array of points.

    Accepts:
    1. A NumPy array already shaped (k, 2) or (2,)     ↦ returned unchanged
    2. An iterable of objects carrying an attribute ``x`` that is array‑like
       length‑2 (e.g. HistoryEntry instances)          ↦ their .x values stacked
    3. An iterable of array‑likes of length‑2           ↦ stacked directly
    """
    if isinstance(path_like, np.ndarray):
        arr = np.atleast_2d(path_like)
        if arr.shape[1] != 2:
            raise ValueError("Path array must have shape (k, 2)")
        return arr.astype(float)

    # Fall‑back: treat as iterable
    coords = []
    for p in path_like:
        # HistoryEntry or similar with '.x'
        if hasattr(p, "x"):
            coords.append(np.asarray(p.x, dtype=float))
        else:
            coords.append(np.asarray(p, dtype=float))
    arr = np.vstack(coords)
    if arr.shape[1] != 2:
        raise ValueError("Each point must be length‑2")
    return arr


# -----------------------------------------------------------------------------#
#   Type aliases
# -----------------------------------------------------------------------------#
Func2D = Callable[[np.ndarray], float]      # accepts shape‑(2,) array, returns scalar


# -----------------------------------------------------------------------------#
#   Internal helpers
# -----------------------------------------------------------------------------#
def _evaluate_on_grid(func: Func2D, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorised helper to evaluate *func* over a mesh grid."""
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    Z = np.array([func(p) for p in pts], dtype=float)
    return Z.reshape(X.shape)


# -----------------------------------------------------------------------------#
#   Contour plot with optional algorithm paths
# -----------------------------------------------------------------------------#
def plot_contours(
    func: Func2D,
    xlim: Tuple[float, float],
    ylim: Tuple[float, float],
    *,
    levels: int | Sequence[float] = 50,
    n_points: int = 400,
    paths: Optional[Sequence[Sequence[np.ndarray]]] = None,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
):
    """
    Draw contour lines of *func* and (optionally) overlay optimisation paths.

    Parameters
    ----------
    func
        A callable ``f(x) -> float`` where ``x`` is a length‑2 NumPy array.
    xlim, ylim
        Tuples giving (min, max) for the two axes.
    levels
        Either an integer specifying the number of contour levels or a
        sequence of level values.
    n_points
        Number of points per axis used to build the evaluation grid.
    paths
        A sequence whose elements are themselves sequences of ``np.ndarray``
        points (the iterates produced by an algorithm).
    labels
        Labels that will appear in the legend for each path.
    ax
        Existing matplotlib Axes to draw on. If *None*, a new figure is made.
    title
        Title for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    # Build evaluation grid
    xs = np.linspace(*xlim, n_points)
    ys = np.linspace(*ylim, n_points)
    X, Y = np.meshgrid(xs, ys)
    Z = _evaluate_on_grid(func, X, Y)

    # Draw contours
    contour_set = ax.contour(X, Y, Z, levels=levels, cmap="viridis")
    ax.clabel(contour_set, inline=1, fontsize=8, fmt="%.2g")

    # Overlay optimisation paths if given
    if paths is not None:
        if labels is None:
            labels = [None] * len(paths)

        markers = ["o", "s", "^", "d", "x", "v", "*", "P"]
        linestyles = ["-", "--", "-.", ":"]
        for i, path in enumerate(paths):
            pts = _to_path_array(path)
            ax.plot(
                pts[:, 0],
                pts[:, 1],
                linestyle=linestyles[i % len(linestyles)],
                marker=markers[i % len(markers)],
                label=labels[i] if i < len(labels) else None,
            )

    # Beautify
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    if title is not None:
        ax.set_title(title)
    if labels is not None and any(lbl is not None for lbl in labels):
        ax.legend()

    return ax


# -----------------------------------------------------------------------------#
#   Convergence curves
# -----------------------------------------------------------------------------#
def plot_convergence(
    histories: Sequence[Sequence["HistoryEntry"]],
    *,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    ylog: bool = True,
    title: str = "Objective value vs. iteration",
):
    """
    Plot objective value against iteration for several optimisation runs.

    Parameters
    ----------
    histories
        Sequence where each element is the ``history`` list returned by an
        ``unconstrainedMinimizer`` instance.
    labels
        Labels for legend entries.
    ax
        Existing matplotlib Axes to draw on. If *None*, a new figure is made.
    ylog
        If *True*, use a logarithmic y‑axis.
    title
        Title for the plot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    if labels is None:
        labels = [None] * len(histories)

    for i, hist in enumerate(histories):
        iters = [entry.k for entry in hist]
        values = [entry.f for entry in hist]
        plot_fn = ax.semilogy if ylog else ax.plot
        plot_fn(iters, values, label=labels[i] if i < len(labels) else None)

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Objective value")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    if labels is not None and any(lbl is not None for lbl in labels):
        ax.legend()

    return ax


__all__ = ["plot_contours", "plot_convergence"]