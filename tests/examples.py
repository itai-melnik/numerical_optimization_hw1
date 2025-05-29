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
  
    
    
    
    return f



    
def rosenbrock(self, x, bool_flag) -> Callable[[np.ndarray, bool], tuple[float, np.ndarray, Optional[np.ndarray]]]: 
    
    pass
    
    
    