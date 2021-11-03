
import os
import pickle
import numpy as np
from exercise_code.networks.linear_model import *

class Loss(object):
    def __init__(self):
        self.grad_history = []
        
    def forward(self,y_out, y_truth):
        return NotImplementedError
    
    def backward(self,y_out, y_truth, upstream_grad=1.):
        return NotImplementedError
    
    def __call__(self,y_out, y_truth):
        loss = self.forward(y_out, y_truth) 
        grad = self.backward(y_out, y_truth)
        return (loss, grad)
    
class L1(Loss):
        
    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of L1 loss for each sample of your training set. 
        """
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the L1 loss.      #
        #########################################################################
        result = np.absolute(y_truth-y_out)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the L1 loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of L1 loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        # hint: you may use np.where here.                                        #
        ###########################################################################
        neg_val=-(y_truth-y_out)
        pos_val=(y_truth-y_out)
        zero_val=(y_truth-y_out)
        grad=(-1)*np.where((y_truth-y_out)<0,-1,1)
        gradient= np.where((y_truth-y_out)==0,0,grad)

        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################        
        return gradient

class MSE(Loss):
  

    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model. 
                y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of MSE loss for each sample of your training set. 
        """    
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the MSE loss.     #
        #########################################################################
       
        dif=y_truth-y_out
        result=np.square(dif)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the MSE loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of MSE loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None
        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        ###########################################################################

        gradient=-2*(y_truth-y_out)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################   
        return gradient    

class BCE(Loss):
  

    def forward(self,y_out, y_truth):
        """
        Performs the forward pass of the binary cross entropy loss function.

        :param y_out: [N, ] array predicted value of your model. 
                y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of binary cross entropy loss for each sample of your training set. 
        """    
        result = None
        #########################################################################
        # TODO:                                                                 #
        # Implement the forward pass and return the output of the BCE loss.     #
        #########################################################################

        sub_yout= np.subtract(1,y_out)
        sub_ytruth=np.subtract(1,y_truth)
        log_y=np.log(1.*y_out)
        sub_log=np.log(1.*sub_yout)
        p1=-(y_truth*log_y)
        p2=(sub_ytruth*sub_log)
        result=np.subtract(p1,p2)
        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################
        
        return result
    
    def backward(self,y_out, y_truth):
        """
        Performs the backward pass of the loss function.

        :param y_out: [N, ] array predicted value of your model. 
               y_truth: [N, ] array ground truth value of your training set. 
        :return: [N, ] array of binary cross entropy loss gradients w.r.t y_out for 
                  each sample of your training set. 
        """
        gradient = None

        ###########################################################################
        # TODO:                                                                   #
        # Implement the backward pass. Return the gradient wrt y_out              #
        ###########################################################################

        sub_o=np.subtract(1,y_out)
        sub_t=np.subtract(1,y_truth)
        #grad_1=sub_t/sub_o
        with np.errstate(divide='ignore', invalid='ignore'):
            c = np.true_divide(sub_t,sub_o)
            c[c == np.inf] = 0
            c = np.nan_to_num(c)
        grad_2=y_truth/y_out
        #inverse=np.linalg.inv(sub_y)
        gradient=np.subtract(c,grad_2)
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################   
        return gradient    
