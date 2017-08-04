import numpy
from math import sqrt


def compute_gradient_RSS_W(H, W, y):
    '''
    This function computes the gradient of RSS(W).
    Args:
        H: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
        y: |N| vector of true labels

    Returns:
        |D| Vector, the gradient of W
    '''
    if type(H) != numpy.matrix:
        H = numpy.matrix(H)
        if H.shape[0] == 1:  # In case there was one feature
            H = H.T
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T  # column vector
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    if type(y) != numpy.matrix:
        y = numpy.matrix(y).T  # column vector
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T
    delta_W = -2 * H.T * (y - H * W)
    return delta_W


def magnitude(W):
    '''
    Computes the magnitude of the vector W
    Args:
        W: an arbitrary sized vector

    Returns:
        magnitude of W
    '''
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    return sqrt(float(W.T * W))


def add_bias_feature(H):
    '''
    Adds an all 1 feature to serve as the bias feature for W[0]

    Args:
        H: |NxD| Matrix of N observations with D features each

    Returns:
        H: |Nx(D+1)| numpy matrix
    '''
    if not type(H) == numpy.matrix:
        H = numpy.matrix(H)
        if H.shape[0] == 1:  # In case there was one feature
            H = H.T
    return numpy.hstack((numpy.ones((H.shape[0], 1)), H))


def graident_descent(H, W, y, step_size, tolerance):
    '''
    Conducts the gradient descent and returns the minimum of the function RSS(W)
    Args:
        H: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features
        y: |N| vector of true labels
        step_size: step size of the gradient
        tolerance: error tolerance
        
    Returns:
        |D| Vector minimizing the RSS
    '''
    if type(W) != numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    H = add_bias_feature(H)
    delta_W = compute_gradient_RSS_W(H, W, y)
    M = magnitude(delta_W)
    iteration = 1
    while M > tolerance:
        if iteration % 500 == 0:
            print('Iteration {0} with Magnitude of {1} for gradiant(RSS(W))'.format(iteration, M))
        W = W - step_size * delta_W
        delta_W = compute_gradient_RSS_W(H, W, y)
        M = magnitude(delta_W)
        iteration += 1
    return W
