import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Callable, Optional





FuncType = Callable[[np.ndarray], Tuple[float, np.ndarray, Optional[np.ndarray]]]  #input: x, output: f(x), f'(x), hessian (optional)

@dataclass
class HistoryEntry:
    k: int
    x: np.ndarray
    f: float
    grad_norm: float

class unconstrainedMinimizer:
    
    """
    Line-search-based minimization for unconstrained problems.
    
    
    parameters:
    f : 
    is the function minimized
    x0 : 
    is the starting point
    max_iter : 
    is the maximum allowed number of iterations
    obj_tol : 
    is the numeric tolerance for successful termination due to small enough objective change or Newton Decrement
    param_tol : 
    is the numeric tolerance for successful termination in terms of small enough distance between iterations
    user_choice:
    "gradient" for gradient descent 
    "newton" for newton method
    
    
    return : the final location, final objective value and a success/failure Boolean flag: 
    success means at least one of the termination criteria is met. Failure means the
    maximal number allowed iterations is reached, or some unexpected termination
    
    
    """
    
    def __init__(self, f: FuncType, x0=np.array((1,1)), obj_tol=1e-12, param_tol=1e-8, max_iter=100, user_choice="gradient", rho=0.5, c1=0.01) -> None:
        
        self.f = f
        self.x = x0.astype(float)
        self.obj_tol = obj_tol
        self.param_tol = param_tol
        self.max_iter = max_iter
        self.user_choice = user_choice.lower() #if user_choice.lower() == "newton" else "gradient"
        
        #constants
        self.rho = rho #backtracking constant
        self.c1 = c1 #Wolfe condition
        
        #previous values
        self.prev_x = None
        self.prev_f_val = None
        
        
        
        #internal storage
        self.history: List[HistoryEntry] = []
        
        
    
    
    def minimize(self) -> Tuple[np.ndarray, float]:
        
        bool_flag = False
       
        need_hessian = True if self.user_choice == 'newton' else False
            
        for k in range(self.max_iter):
            
            f_val, g, h = self.f(self.x, need_hessian)
            
            
            #save current iteration
            self._save_history(k, f_val, g)
            
            #print to console:
            # print('iteration number:',k + 1 )
            # print('current location 𝑥𝑖:',self.x)
            # print('current objective value 𝑓(𝑥𝑖 ):',f_val )
            
            #break if stopping criteria met
            if self._is_converged(k):
                bool_flag = True
                break
            
            
            #compute p_k(the direction) changes depends on which method we use
            p = self._compute_direction(g, h)
            
            
            #computer alpha_k
            alpha = self.backtracking(self.x, f_val, g, p )
            
            #update
            x_new = self.x + alpha * p
            
            
            #store previous values for stopping termination conditions 
            
            self.prev_x = self.x
            self.prev_f_val = f_val
            
    
            #x_(k+1) = x_new
            self.x = x_new
            
              
        
        #returns final location, final value and bool flag
        print('Method:', self.user_choice)
        print('iteration number:',k)
        print('current location 𝑥𝑖:',self.x)
        print('current objective value 𝑓(𝑥𝑖 ):',f_val )
        print('output flag:', bool_flag)
        return self.x, self.history[-1].f, bool_flag
            
            
            
    ### helper methods ###
    
    def _is_converged(self, k: int) -> bool:
        """check the stopping criteria"""
        
        if k == 0:
            return False
        
        param_change = np.linalg.norm(self.x - self.prev_x)
        obj_change = abs(self.history[-1].f - self.prev_f_val)
        
        
        if(param_change < self.param_tol or obj_change < self.obj_tol):
            return True
        
        return False
        
    
    
    def _compute_direction(self, g, h) -> np.ndarray:
        """compute pk"""
        
        #for gradient descent 
        if(self.user_choice=='gradient'):
            #p_k = -g
            return -g 
        
        
        #for newton method
        elif (self.user_choice=='newton'):
            try:
                return -np.linalg.solve(h, g) #solves ax = b (gets x) h*p = -g since p = -g/h so we get p. AVOID computing inverse
            except np.linalg.LinAlgError:
                return -g
        
        
              
            
         
    
    def backtracking(self, x, f_val, g, p) -> float:
        """ used to compute alpha (the step size)"""
        
        alpha = 1.0 #initial value
        
        while self.f(x + alpha*p)[0] > f_val + self.c1*alpha*g.dot(p):
            alpha *= self.rho
            
            if alpha < 1e-12:
                break
            
            
        return alpha
        
    
    def _save_history(self, k, f_val, g) -> None:
        """save the history from previous iterations"""
        self.history.append(HistoryEntry(k=k,x=self.x.copy(),f=f_val,grad_norm=np.linalg.norm(g)))
        
        
        
    def get_history(self) -> List[HistoryEntry]:
        """ return the iteration history"""
        return self.history
        
        
        
    
        
            
    
        
        