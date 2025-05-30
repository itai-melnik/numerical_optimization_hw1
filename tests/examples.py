from typing import Callable, Optional
import numpy as np




def make_quadratic(Q: np.ndarray) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]:
    """
    Return a callable f(x, need_hessian=False) for the quadratic form xᵀ Q x.

    Parameters
    ----------
    Q : np.ndarray
        Symmetric positive-definite matrix defining the quadratic form.

    Returns
    -------
    f : Callable
        A function that computes value, gradient and (optionally) Hessian.
    """
    
    
    hess_const = 2.0 * Q #constant hessian since it is quadratic 
    
    
    def _quad(x: np.ndarray, need_hessian: bool = False):
        """Quadratic form value, gradient and (optionally) Hessian.
        """
        
        # Value  xᵀ Q x  -> scalar
        val = x.T @ Q @ x 
        
        # Gradient 2 Q x  -> vector with same shape as x
        grad = 2.0 * Q @ x
        
         # Return Hessian only when explicitly requested
        hessian = hess_const if need_hessian else None
        
        return val, grad, hessian
        
        
        
    return _quad


def Qi():
    Qi = np.diag([1,1])
    return make_quadratic(Qi)

def Qii():
    Qii = np.diag([1,100])
    return make_quadratic(Qii)

def Qiii():
    q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
    Qiii = q.T  @ Qii @ q
    return make_quadratic(Qiii)





  
def rosenbrock(x: np.ndarray, bool_flag: bool = False) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]: 
    
    val = 100 * ((x[1]- x[0] ** 2) ** 2) + (1 - x[0])**0
    
    
    grad0 = -400 * (x[1] - x[0] ** 2) * (x[0]) -2 * (1- x[0])
    grad1 = 200 * (x[1] - x[0] ** 2)
    grad = np.array([grad0, grad1])
    
    #d^2x0
    h0 = 1200 * x[0] - 400 * x[1] + 2 
    
    #d^2x1
    h1 = 200  
    
    #dx0dx1                 
    h2 = -400 * x[0] 
    hessian = np.array([[h0, h2], [h2, h1]]) if bool_flag else None
    
    return val, grad, hessian


def linear_func(x: np.ndarray, a: np.ndarray, bool_flag: bool = False) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]:
    
    #check if same shape
    
    val = a.T @ x
    
    grad = a
    
    hessian = 0 #check this needs to zero hessian
    
    return val, grad, hessian




def triangle_func(x: np.ndarray, bool_flag: bool = False) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]: 
    pass
    
    
    
    