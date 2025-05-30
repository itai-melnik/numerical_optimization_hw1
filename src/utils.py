"""
Utility helpers for visualising 2-D optimisation problems and their progress.

Functions
---------
plot_contours(func, xlim, ylim, *, levels=50, n_points=400,
              paths=None, labels=None, ax=None, title=None)
    Draws contour lines of a 2-D objective function and (optionally) the
    optimisation paths produced by one or more algorithms.

plot_convergence(histories, *, labels=None, ax=None,
                 ylog=True, title='Objective value vs. iteration')
    Plots f(x_k) versus iteration k for several optimisation methods on a
    single set of axes so their rates of decrease can be compared.
"""
from __future__ import annotations

from typing import Callable, Iterable, Sequence, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt

Func2D = Callable[[np.ndarray], float]      # accepts shape-(2,) array, returns scalar


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
    paths: Optional[Iterable[np.ndarray]] = None,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
) -> plt.Axes:
    """
    Parameters
    ----------
    func
        Callable that maps a 2-D point (numpy array of length 2) to a scalar.
    xlim, ylim
        (min, max) pairs that delimit the rectangle to show.
    levels
        Passed straight to `plt.contour`.  Either an int (#levels) or explicit
        contour values.
    n_points
        Grid resolution per axis.
    paths
        Iterable of 2-D arrays of shape (k_i, 2).  Each row is the x-coordinate
        visited by an algorithm.
    labels
        Legend labels matching `paths`.  Ignored if `paths is None`.
    ax
        Existing axes to draw on; if omitted a new figure/axes is created.
    title
        Title for the figure.  If omitted it is derived from `func.__name__`.
    """
    if ax is None:  # fresh figure
        fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)

    # mesh grid
    xs = np.linspace(*xlim, n_points)
    ys = np.linspace(*ylim, n_points)
    X, Y = np.meshgrid(xs, ys)
    Z = np.empty_like(X)

    # evaluate func on grid (vectorised loop to spare memory)
    for i in range(n_points):
        for j in range(n_points):
            Z[i, j] = func(np.array([X[i, j], Y[i, j]]))

    # draw contours
    cs = ax.contour(X, Y, Z, levels=levels, linewidths=0.8, cmap="viridis")
    ax.clabel(cs, inline=True, fontsize=8)

    # optional algorithm paths
    if paths is not None:
        if labels is None:
            labels = [f"Path {i}" for i in range(len(paths))]
        for path, label in zip(paths, labels):
            ax.plot(path[:, 0], path[:, 1], marker="o", markersize=3, lw=1.5,
                    label=label)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_aspect("equal", adjustable="box")

    if title is None:
        title = f"Contours of {func.__name__}"
    ax.set_title(title)

    if paths is not None:
        ax.legend(loc="best")

    return ax


# -----------------------------------------------------------------------------#
#   Convergence plot
# -----------------------------------------------------------------------------#
def plot_convergence(
    histories: Sequence[Sequence[float]],
    *,
    labels: Optional[Sequence[str]] = None,
    ax: Optional[plt.Axes] = None,
    ylog: bool = True,
    title: str = "Objective value vs. iteration",
) -> plt.Axes:
    """
    Parameters
    ----------
    histories
        Each element is a list/array of objective values f(x_k) in order of
        iteration for one optimisation run.
    labels
        Legend labels matching each history list.
    ax
        Existing axes to draw on; if omitted a new figure/axes is created.
    ylog
        If True, use a log scale for the y-axis (helps compare rates).
    title
        Figure title.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)

    if labels is None:
        labels = [f"Run {i}" for i in range(len(histories))]

    for fvals, label in zip(histories, labels):
        ax.plot(range(len(fvals)), fvals, marker="o", markersize=3,
                linewidth=1.5, label=label)

    ax.set_xlabel("Iteration $k$")
    ax.set_ylabel(r"$f(x_k)$")
    if ylog:
        ax.set_yscale("log")

    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    return ax