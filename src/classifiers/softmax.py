import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  
  loss = 0.0
  dW = np.zeros(W.shape, dtype='float')
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  #############################################################################
  # Compute the softmax loss and its gradient using explicit loops.           #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  for i in xrange(num_train):
      
      scores = X[i].dot(W)
      scores -= np.max(scores) # numeric stability
      loss += -np.log(np.exp(scores[y[i]])/np.sum(np.exp(scores)))
      
      for j in xrange(num_classes):
          softmax = np.exp(scores[j])/np.sum(np.exp(scores))
          if j == y[i]:
              dW[:, j] += (softmax - 1) * X[i]
          else:
              dW[:, j] += softmax * X[i]

  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  dW /= num_train
  dW += reg * W
  
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
    
  #analytical way of computing loss and gradiant
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros(W.shape, dtype='float')
  num_train = X.shape[0]
  num_classes = W.shape[1]
  
  #############################################################################
  # Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  S = X.dot(W)
  S -= np.max(S, axis=1)[:, np.newaxis]
  num = np.exp(S[np.arange(len(S)), y])
  den = np.sum(np.exp(S), axis=1)
  loss = -np.log(num/den)
  loss = np.sum(loss)
  
  
  loss /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  
  S = np.exp(S)
  den = np.sum(S, axis=1)[:, np.newaxis]
  S = S / den
  S[np.arange(len(S)), y] -= 1 
  
  dW = X.T.dot(S)    
  dW /= num_train
  dW += reg * W
      
  return loss, dW

  '''
  # Initialization
  loss = 0.0
  dW = np.zeros(W.shape, dtype='float')
  N = X.shape[0]
  C = W.shape[1]
  
  # Forward Pass
  scores = X.dot(W) # scores - N X C
  scores_stable = scores - np.max(scores, axis=1)[:, np.newaxis] # scores_stable - N X C
  scores_exp = np.exp(scores_stable) # scores_exp - N X C
  scores_sum_inv = 1 / np.sum(scores_exp, axis=1)[:, np.newaxis] # scores_sum_inv - N X 1
  probs = scores_exp * scores_sum_inv # probs - N X C
  correct_class_probs = probs[ np.arange(N) , y] # correct_class_probs - 1 X N
  log_probs = np.log(correct_class_probs) # log_probs - 1 X N
  sum_log_probs = np.sum(log_probs) # sum_log_probs - scalar - 
  
  data_loss = -sum_log_probs / N # data_loss - scalar -
  reg_loss = 0.5 * reg * np.sum(W * W) # reg_loss - scalar -
  
  loss = data_loss + reg_loss # total_loss - scalar -
  
  # Backward Pass
  d_loss = 1.0
  d_data_loss = 1.0 * d_loss
  d_reg_loss = 1.0 * d_loss
  
  d_sum_log_probs = -1/N
  d_log_probs = np.ones(N) * (-1/N)
  
  dW += reg * W
  
  return loss, dW
  '''
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  