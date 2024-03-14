import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict

from scipy.optimize.linesearch import scalar_search_wolfe2
from scipy.linalg import cho_factor, cho_solve
import time


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Armijo':
            if previous_alpha is None:
                alpha = self.alpha_0
            else:
                alpha = previous_alpha
            phi_zero = oracle.func(x_k)
            phi_zero_derivative = oracle.grad(x_k) @ d_k
            
            while phi_zero + self.c1 * alpha * phi_zero_derivative < oracle.func(x_k + alpha * d_k):
                alpha /= 2
                
            return alpha
        
        elif self._method == 'Constant':
            return self.c
        
        elif self._method == 'Wolfe':
            phi = lambda a: oracle.func_directional(x_k, d_k, a)
            phi_derivative = lambda a: oracle.grad_directional(x_k, d_k, a)
            
            phi_zero = oracle.func(x_k)
            phi_zero_derivative = oracle.grad(x_k) @ d_k
            
            result, _, _, _ = scalar_search_wolfe2(phi, phi_derivative, phi_zero, None, phi_zero_derivative, self.c1, self.c2)
 
            if result is not None:
                return result
            else:
                if previous_alpha is None:
                    alpha = self.alpha_0
                else:
                    alpha = previous_alpha
                phi_zero = oracle.func(x_k)
                phi_zero_derivative = oracle.grad(x_k) @ d_k

                while phi_zero + self.c1 * alpha * phi_zero_derivative < oracle.func(x_k + alpha * d_k):
                    alpha /= 2

                return alpha

        return None


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

def add_new_step(history, x, function, gradient, seconds):
    history['x'].append(np.copy(x))
    history['func'].append(function)
    history['grad_norm'].append(np.linalg.norm(gradient))
    history['time'].append(seconds)

def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    
    if trace:
        history = defaultdict(list) 
    else:
        history = None
        
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    time_0 = time.time()
    grad_0 = oracle.grad(x_0)
    grad_k = oracle.grad(x_k)
    f_k = oracle.func(x_k)
    time_k = time.time() - time_0
 
    if trace:
        add_new_step(history, x_k, f_k, grad_k, time_k)
        
    if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(grad_k))):
        return x_k, 'computational_error', history
    
    if tolerance * np.linalg.norm(grad_0) ** 2 >= np.linalg.norm(grad_k) ** 2:
        return x_k, 'success', history
    
    try:
        alpha_k = line_search_tool.alpha_0
    except AttributeError:
        alpha_k = 1.0
 
    for k in range(max_iter):
        alpha_k = line_search_tool.line_search(oracle, x_k, -grad_k, alpha_k)
        x_k -= alpha_k * grad_k
        grad_k = oracle.grad(x_k)
        f_k = oracle.func(x_k)
        time_k = time.time() - time_0
 
        if trace:
            add_new_step(history, x_k, f_k, grad_k, time_k)
            
        if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(grad_k))):
            return x_k, 'computational_error', history
        
        if tolerance * np.linalg.norm(grad_0) ** 2 >= np.linalg.norm(grad_k) ** 2:
            return x_k, 'success', history
 
    return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    if trace:
        history = defaultdict(list) 
    else:
        history = None
        
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)

    time_0 = time.time()
    grad_0 = oracle.grad(x_0)
    grad_k = oracle.grad(x_k)
    f_k = oracle.func(x_k)
    time_k = time.time() - time_0
    alpha_0 = 1.0
 
    if trace:
        add_new_step(history, x_k, f_k, grad_k, time_k)
        
    if tolerance * np.linalg.norm(grad_0) ** 2 >= np.linalg.norm(grad_k) ** 2:
        return x_k, 'success', history
    
    if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(grad_k))):
        return x_k, 'computational_error', history
    
    try:
        hess = oracle.hess(x_k)
        grad = oracle.grad(x_k)
        hess_c, hess_low = cho_factor(hess)
        d_k = cho_solve((hess_c, hess_low), -grad)
    except LinAlgError:
        return x_k, 'newton_direction_error', history
 
    for k in range(max_iter):
        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, alpha_0)
        x_k += alpha_k * d_k
        grad_k = oracle.grad(x_k)
        f_k = oracle.func(x_k)
        time_k = time.time() - time_0
 
        if trace:
            add_new_step(history, x_k, f_k, grad_k, time_k)
            
        if not (np.all(np.isfinite(x_k)) and np.all(np.isfinite(d_k))):
            return x_k, 'computational_error', history
        
        if tolerance * np.linalg.norm(grad_0) ** 2 >= np.linalg.norm(grad_k) ** 2:
            return x_k, 'success', history
        
        try:
            hess = oracle.hess(x_k)
            grad = oracle.grad(x_k)
            hess_c, hess_low = cho_factor(hess)
            d_k = cho_solve((hess_c, hess_low), -grad)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
 
    return x_k, 'iterations_exceeded', history
