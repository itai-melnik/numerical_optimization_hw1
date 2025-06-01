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
    
    
    def _quad(x: np.ndarray, bool_flag: bool = False):
        """Quadratic form value, gradient and (optionally) Hessian.
        """
        
        # Value  xᵀ Q x  -> scalar
        val = x.T @ Q @ x 
        
        # Gradient 2 Q x  -> vector with same shape as x
        grad = 2.0 * Q @ x
        
         # Return Hessian only when explicitly requested
        hessian = hess_const if bool_flag else None
        
        return val, grad, hessian
        
        
        
    return _quad

Qi = np.diag([1,1])

quad_i = make_quadratic(Qi)




Qii = np.diag([1,100])
quad_ii = make_quadratic(Qii)

q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])
Qiii = q.T  @ Qii @ q
quad_iii = make_quadratic(Qiii)





  
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


def linear_func(x: np.ndarray, bool_flag: bool = False, a: np.ndarray = None ) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]:
    
    #check if same shape
    if a is None:
        a = np.ones_like(x)
    
    val = float(a.T @ x)
    
    grad = a.copy()
    
    hessian = np.zeros((x.size, x.size)) if bool_flag else None
    
    return val, grad, hessian




def triangle_func(x: np.ndarray, bool_flag: bool = False) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]: 
    
    x1, x2 = x
    
    # Pre‑compute the three exponentials
    a = np.exp(x1 + 3.0 * x2 - 0.1)
    b = np.exp(x1 - 3.0 * x2 - 0.1)
    c = np.exp(-x1 - 0.1)
    
    val = float(a + b + c)
    
    grad = np.array([
        a + b - c,         # ∂f/∂x₁
        3.0 * a - 3.0 * b  # ∂f/∂x₂
    ])
    
    
    hessian = None
    if bool_flag:
        hxx = a + b + c              # ∂²f/∂x₁²
        hxy = 3.0 * a - 3.0 * b      # ∂²f/∂x₁∂x₂ (symmetric)
        hyy = 9.0 * a + 9.0 * b      # ∂²f/∂x₂²
        hessian = np.array([[hxx, hxy], [hxy, hyy]])
        
    return val, grad, hessian

    
    
    