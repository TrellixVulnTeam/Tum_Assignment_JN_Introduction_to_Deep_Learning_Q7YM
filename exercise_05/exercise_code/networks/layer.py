import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    :param x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    :param w: A numpy array of weights, of shape (D, M)
    :param b: A numpy array of biases, of shape (M,)

    :return out: output, of shape (N, M)
    :return cache: (x, w, b)
    """
    out = None
    ########################################################################
    # TODO: Implement the affine forward pass. Store the result in out.    #
    # You will need to reshape the input into rows.                        #
    ########################################################################
    #vector=[]
    #for i in range(len(x)):
     #   vector.append(np.dot((x[i].flatten()),w)+b)
    
    #out=vector
    #out=np.dot(vec,w)
    #out=np.sum(product,b)
    out = x.reshape(x.shape[0], w.shape[0]).dot(w) + b           
                   
        

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    :param dout: Upstream derivative, of shape (N, M)
    :param cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: A numpy array of biases, of shape (M,

    :return dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    :return dw: Gradient with respect to w, of shape (D, M)
    :return db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ########################################################################
    # TODO: Implement the affine backward pass.                            #
    # Hint: Don't forget to average the gradients dw and db                #
    ########################################################################

    #for i in range(len(x)):
    dx = dout.dot(w.T).reshape(x.shape)
    dw = (x.reshape(x.shape[0], w.shape[0]).T.dot(dout))/x.shape[0]
    #dw=np.mean(dw,axis=0,keepdims=True)
    db = (np.sum(dout, axis=0))/x.shape[0] 
      

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx, dw, db


def sigmoid_forward(x):
    """
    Computes the forward pass for a layer of sigmoids.

    :param x: Inputs, of any shape

    :return out: Output, of the same shape as x
    :return cache: x
    """
    out = None
    ########################################################################
    # TODO: Implement the Sigmoid forward pass.                            #
    ########################################################################

    out=1/(1+np.exp(-x))

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    cache = out
    return out, cache


def sigmoid_backward(dout, cache):
    """
    Computes the backward pass for a layer of sigmoids.

    :param dout: Upstream derivatives, of any shape
    :param cache: Input x, of same shape as dout

    :return dx: Gradient with respect to x
    """
    dx = None
    y = cache
    ########################################################################
    # TODO: Implement the Sigmoid backward pass.                           #
    ########################################################################
    #dx=np.exp(-y)/((1+np.exp(-y))*(1+np.exp(-y)))
    #dx=np.multiply(sigma,(1-sigma))
    func=y*(1-y)
    dx=dout*func

    ########################################################################
    #                           END OF YOUR CODE                           #
    ########################################################################
    return dx
