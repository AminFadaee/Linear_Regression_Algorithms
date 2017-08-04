import numpy


def compute_weights(H, y):
    '''
    This function compute the regression coefficient using the Closed-form solution.
    
    Args:
        H: |NxD| Matrix of N observations with D features each
        y: |N| Vector of labels

    Returns:
        W: |D| Vector of features
    '''
    if not type(H) == numpy.matrix:
        H = numpy.matrix(H)
        if H.shape[0] == 1:  # In case there was one feature
            H = H.T
    if not type(y) == numpy.matrix:
        y = numpy.matrix(y).T
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T
    H = add_bias_feature(H)
    first_part = (H.T * H)

    if not (first_part.shape[0] == first_part.shape[1] and numpy.linalg.matrix_rank(first_part) == first_part.shape[0]):
        raise Exception("'H.T x H' is not invertible!")

    W = numpy.linalg.inv(first_part) * H.T * y
    return W


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


def predict(H, W):
    '''
    Predicts for H the labels, based on W 
    Args:
        H: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features

    Returns:
        y: |N| Vector of predicted labels
    '''
    H = add_bias_feature(H)
    if not type(W) == numpy.matrix:
        W = numpy.matrix(W).T
    elif W.shape[0] == 1:  # creating a column vector
        W = W.T
    y = H * W
    return y


def RSS(y, H, W):
    '''
    Computes the residual sum of squares.
    Args:
        y: |N| vector of true labels
        H: |NxD| Matrix of N observations with D features each
        W: |D| Vector of features

    Returns:
        y: |N| Vector of labels
    '''
    if not type(y) == numpy.matrix:
        y = numpy.matrix(y).T
    elif y.shape[0] == 1:  # creating a column vector
        y = y.T

    E = y - predict(H, W)
    return (E * E.T).sum()
