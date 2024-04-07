import numpy as np
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool

import scipy
from datetime import datetime

def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if trace:
        history = defaultdict(list)
    else:
        history = None
    x_k = np.copy(x_0).astype(np.float64)
    Ax_minus_b = matvec(x_k) - b
    der = -Ax_minus_b
    treshold = tolerance * scipy.linalg.norm(b)

    start_time = datetime.now()
    found = False

    if max_iter is None:
        max_iter = x_k.size

    for i in range(max_iter + 1):
        Ax_minus_b_norm = scipy.linalg.norm(Ax_minus_b)

        if trace:
            delta = datetime.now() - start_time
            history['time'].append(delta.seconds + delta.microseconds * 1e-6)
            history['residual_norm'].append(np.copy(Ax_minus_b_norm))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

        if display:
            print('step', history['time'][-1] if history else '')

        if Ax_minus_b_norm <= treshold:
            found = True
            break

        if i == max_iter:
            break

        A_der = matvec(der)
        Ax_minus_b_sq = np.dot(Ax_minus_b, Ax_minus_b)
        x_k += Ax_minus_b_sq / np.dot(der, A_der) * der
        Ax_minus_b += Ax_minus_b_sq / np.dot(der, A_der) * A_der
        der = -Ax_minus_b + np.dot(Ax_minus_b, Ax_minus_b) / Ax_minus_b_sq * der

    if found:
        return x_k, 'success', history
    
    return x_k, 'iterations_exceeded', history


def update_history(trace, display, history, delta, f, gradient_norm, x_k):
    if trace:
        history['time'].append(delta.seconds + delta.microseconds * 1e-6)
        history['func'].append(np.copy(f))
        history['grad_norm'].append(np.copy(gradient_norm))
        if x_k.size <= 2:
            history['x'].append(np.copy(x_k))

    if display:
        print('step', history['time'][-1] if history else '')


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if trace:
        history = defaultdict(list)
    else:
        history = None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    start_time = datetime.now()
    found = False
    alpha = None

    grads = deque()
    grad_diffs = deque()
    gradient = oracle.grad(x_k)
    treshold = None

    for iteration in range(max_iter + 1):
        f = oracle.func(x_k)
        gradient_norm = scipy.linalg.norm(gradient)
        if treshold is None:
            treshold = np.sqrt(tolerance) * gradient_norm
            
        delta = datetime.now() - start_time
        update_history(trace, display, history, delta, f, gradient_norm, x_k)
            
        if gradient_norm <= treshold:
            found = True
            break
            
        if iteration == max_iter:
            break
            
        der = -gradient
        if grads:
            mus = []
            for grad, diff in zip(reversed(grads), reversed(grad_diffs)):
                mu = np.dot(grad, der) / np.dot(grad, diff)
                mus.append(mu)
                der -= mu * diff
            der *= np.dot(grads[-1], grad_diffs[-1]) / np.dot(grad_diffs[-1], grad_diffs[-1])
            for grad, diff, mu in zip(grads, grad_diffs, reversed(mus)):
                beta = np.dot(diff, der) / np.dot(grad, diff)
                der += (mu - beta) * grad
                
        if alpha is not None:
            alpha = line_search_tool.line_search(oracle, x_k, der, 2.0 * alpha)
        else:
            alpha = line_search_tool.line_search(oracle, x_k, der, None)
            
        x_k += alpha * der
        last_gradient = np.copy(gradient)
        gradient = oracle.grad(x_k)

        if memory_size > 0:
            if len(grads) == memory_size:
                grads.popleft()
                grad_diffs.popleft()
            grads.append(alpha * der)
            grad_diffs.append(gradient - last_gradient)

    if found:
        return x_k, 'success', history
    
    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if trace:
        history = defaultdict(list)
    else:
        history = None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0).astype(np.float64)

    start_time = datetime.now()
    found = False
    treshold = None
    for iteration in range(max_iter + 1):
        f = oracle.func(x_k)
        gradient = oracle.grad(x_k)
        gradient_norm = scipy.linalg.norm(gradient)
        if treshold is None:
            treshold = np.sqrt(tolerance) * gradient_norm

        delta = datetime.now() - start_time
        update_history(trace, display, history, delta, f, gradient_norm, x_k)

        if gradient_norm <= treshold:
            found = True
            break

        if iteration == max_iter:
            break

        e = min(0.5, np.sqrt(gradient_norm))
        gradient_found = False

        while not gradient_found:
            cg, _, _ = conjugate_gradients(lambda y: oracle.hess_vec(x_k, y), -gradient, -gradient, tolerance=e)
            e /= 10
            if np.dot(cg, gradient) < 0:
                gradient_found = True

        alpha = line_search_tool.line_search(oracle, x_k, cg, 1.0)
        x_k += alpha * cg

    if found:
        return x_k, 'success', history
    
    return x_k, 'iterations_exceeded', history