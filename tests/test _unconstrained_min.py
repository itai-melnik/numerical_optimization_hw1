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
    
    
    #recommended starting points
    x0 = np.array([1,1])
    x0_r = np.array([-1,2])
    
    
    #recommended max iterations
    max_iter = 100
    max_iter_r = 10000
    
    
    #recommended constants
    rho = 0.01 
    c0 = 0.5
    
    
   
    def test_Qi(self):
        
        gd = unconstrainedMinimizer(
            quad_i,
            x0=self.x0,
            user_choice="gradient",               # gradient descent
        )
        
        newton = unconstrainedMinimizer(
            quad_i,
            x0=self.x0,
            user_choice="newton",               
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
        
        
        
        
        
        plot_contours(
            lambda x: quad_i(x)[0],
            xlim=(-0.5, 1.1),
            ylim=(-0.5, 1.1),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"Qi GD vs Newton path",
        )
        
        
    
    def test_Qii(self):
        
        gd = unconstrainedMinimizer(
            quad_ii,
            x0=self.x0,
            user_choice="gradient",               # gradient descent
        )
        
        newton = unconstrainedMinimizer(
            quad_ii,
            x0=self.x0,
            user_choice="newton",               
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
        
        
    
        plot_contours(
            lambda x: quad_ii(x)[0],
            xlim=(-0.5, 1.1),
            ylim=(-0.5, 1.1),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"Qii GD vs Newton path",
        )
        
        
        
        
    def test_Qiii(self):
        
        gd = unconstrainedMinimizer(
            quad_iii,
            x0=self.x0,
            user_choice="gradient",               # gradient descent
        )
        
        newton = unconstrainedMinimizer(
            quad_iii,
            x0=self.x0,
            user_choice="newton",               
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
        
        
    
        plot_contours(
            lambda x: quad_iii(x)[0],
            xlim=(-2, 1.5),
            ylim=(-2, 1.5),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"Qiii GD vs Newton path",
        )
        
        
        
        
    def test_rosenbrock(self):
        
        gd = unconstrainedMinimizer(
            rosenbrock,
            x0=self.x0_r,
            user_choice="gradient",
            max_iter=self.max_iter_r               
        )
        
        newton = unconstrainedMinimizer(
            rosenbrock,
            x0=self.x0_r,
            user_choice="newton",
            max_iter=self.max_iter_r              
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
        
    
        plot_contours(
            lambda x: rosenbrock(x)[0],
            xlim=(-2,0),
            ylim=(0,3),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"Rosenbrock GD vs Newton path",
        )
        
        
        
        
        
        
    def test_linear(self):
        
        gd = unconstrainedMinimizer(
            linear_func,
            x0=self.x0,
            user_choice="gradient",              
        )
        
        newton = unconstrainedMinimizer(
            linear_func,
            x0=self.x0,
            user_choice="newton",             
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
        
    
        plot_contours(
            lambda x: linear_func(x=x, a=np.array([1,2]))[0],
            xlim=(-100,100),
            ylim=(-100,100),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"linear function GD vs Newton path",
        )
        
        
        
        
        
    def test_triangle(self):
        
        gd = unconstrainedMinimizer(
            linear_func,
            x0=self.x0,
            user_choice="gradient",              
        )
        
        newton = unconstrainedMinimizer(
            linear_func,
            x0=self.x0,
            user_choice="newton",            
        )
        
        gd_x_star, gd_f_star, gd_success = gd.minimize()
        gd_hist = gd.get_history()
        
        newton_x_star, newton_f_star, newton_success = newton.minimize()
        newton_hist = newton.get_history()
    
        plot_contours(
            lambda x: linear_func(x=x)[0],
            xlim=(-100,100),
            ylim=(-100,100),
            paths=[gd_hist, newton_hist],                         # pass HistoryEntry list directly
            labels=["GD", "NEWTON"],
            title=r"Corner Triangle GD vs Newton path",
        )
        
        print(gd_success)
        
        plt.show()
        
        
        

           
if __name__ == '__main__':
    unittest.main() 