import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict
import time

from utils import get_line_search_tool

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
