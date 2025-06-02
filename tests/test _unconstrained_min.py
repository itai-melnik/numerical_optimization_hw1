# --- makes project root importable ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.unconstrained_min import unconstrainedMinimizer
from tests.examples import * 
from src.utils import plot_contours, plot_convergence


class TestUnconstrainedMin(unittest.TestCase):
    
    list_final_output = []
    #recommended starting points
    x0 = np.array([1,1])
    x0_r = np.array([-1,2])
    
    
    #recommended max iterations
    max_iter = 100
    max_iter_r = 10000
    
    
    #recommended constants
    rho = 0.01 
    c0 = 0.5

    # ------------------------------------------------------------------
    # Helper func
    # ------------------------------------------------------------------
    def _run_case(
        self,
        func,
        *,
        x0: np.ndarray,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        title: str,
        contour_func=None,
        max_iter: int | None = None,
    ) -> None:
        """Run GD & Newton on *func*, then plot contours & convergence."""
        # build optimisers
        gd = unconstrainedMinimizer(
            func, x0=x0, user_choice="gradient", max_iter=max_iter or self.max_iter
        )
        newton = unconstrainedMinimizer(
            func, x0=x0, user_choice="newton", max_iter=max_iter or self.max_iter
        )

        # optimise
        gd.minimize();  newton.minimize()
        gd_hist = gd.get_history();  newton_hist = newton.get_history()

        # choose which callable produces scalar value for contours
        cfunc = contour_func if contour_func is not None else (lambda v: func(v)[0])

        plot_contours(
            cfunc,
            xlim=xlim, ylim=ylim,
            paths=[gd_hist, newton_hist],
            labels=["GD", "NEWTON"],
            title=title,
        )
        plot_convergence([gd_hist, newton_hist], labels=["GD", "NEWTON"])
    
    def test_Qi(self):
        self._run_case(
            quad_i,
            x0=self.x0,
            xlim=(-0.5, 1.1), ylim=(-0.5, 1.1),
            title="Qi GD vs Newton path",
        )

    def test_Qii(self):
        self._run_case(
            quad_ii,
            x0=self.x0,
            xlim=(-0.5, 1.1), ylim=(-0.5, 1.1),
            title="Qii GD vs Newton path",
        )

    def test_Qiii(self):
        self._run_case(
            quad_iii,
            x0=self.x0,
            xlim=(-2, 1.5), ylim=(-2, 1.5),
            title="Qiii GD vs Newton path",
        )

    def test_rosenbrock(self):
        self._run_case(
            rosenbrock,
            x0=self.x0_r,
            xlim=(-2,2 ), ylim=(-1, 2.5),
            title="Rosenbrock GD vs Newton path",
            max_iter=self.max_iter_r,
        )

    def test_linear(self):
        self._run_case(
            linear_func,
            x0=self.x0,
            xlim=(-100, 100), ylim=(-100, 100),
            title="Linear function GD vs Newton path",
            contour_func=lambda x: linear_func(x=x, a=np.array([1, 2]))[0],
        )

    def test_triangle(self):
        self._run_case(
            triangle_func,
            x0=self.x0,
            xlim=(-1, 1.5), ylim=(-0.5, 1.5),
            title="Corner Triangle GD vs Newton path",
            contour_func=lambda x: triangle_func(x)[0],
        )
    
        plt.show()
           

if __name__ == '__main__':
    unittest.main() 