import torch
import torch.nn.functional as F
from torch import nn


"""
-----------------------------------------------------------------
WEIGHT ONE HOT ACTIVATION

A weight constrainer which:
    - in forward pass: keeps only the weight with highest absolute value and set others to zero
    - in backpropagation: gradient passes through this function using STE (Straight-Through Estimator)
"""

class getMaxFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        ctx.save_for_backward(w)
        w = w.clone()
        max_pos = torch.argmax(w.abs(), dim=0)
        new_w = torch.zeros_like(w)
        new_w[max_pos, torch.arange(w.size(1))] = w[max_pos, torch.arange(w.size(1))]
        return new_w

    @staticmethod
    def backward(ctx, grad_output): #STE V1
        w, = ctx.saved_tensors #keep in mind w is not affected during backprop
        grad_input = grad_output.clone()
        return grad_input


class getMax(nn.Module):
    def forward(self, w):
        return getMaxFunction.apply(w)

"""
--------------------------------------------------------------------------------------------
POSITIVE RULE'S WEIGHT

forces weights to be positive, using ReLU, but in backpropagation
it uses a LeakyReLU to allow the gradient to move also for zero
valued weights
"""

class forcePositiveFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, slope=0.01):
        ctx.save_for_backward(w)
        ctx.slope = slope
        return torch.clamp(w, min=0)

    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        slope = ctx.slope
        if slope is not None:
            grad_input = torch.where(w > 0, grad_output, slope * grad_output)
        else:
            grad_input = (w > 0) * grad_output
        return grad_input

class forcePositive(nn.Module):
    def __init__(self, slope=0.01):
        super(forcePositive, self).__init__()
        self.slope = slope
    def forward(self, w):
        return forcePositiveFunction.apply(w, self.slope)


"""
------------------------------------------------------------------------------------------
SIGN ACTIVATION FUNCTION

This is a sign activation function which uses STE estimator
during backpropagation
"""

class SignFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        new_x = torch.sign(x)
        return new_x

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs, None

class Sign(nn.Module):
    def forward(self, x):
        return SignFunction.apply(x)


"""
----------------------------------------------------------------------------------------------
----------------- NOT WORKING/NOT TESTED FUNCTIONS--------------------------------------------
----------------------------------------------------------------------------------------------
"""



"""
SOFTMAX RULE'S WEIGHT <-- not sure if it works

defines each rule's weight, with the constraint of being
positive and summing up to one (s.t. each corresponds to a
percentage contribution)

"""

class ruleWeightFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w):
        new_w = nn.functional.softmax(w, dim=-1)
        ctx.save_for_backward(new_w)
        return new_w

    @staticmethod
    def backward(ctx, grad_output):
        new_w, = ctx.saved_tensors
        batch_size = new_w.shape[0]
        num_weights = new_w.shape[1]

        diag = new_w * (1 - new_w)

        jacobian_off_diag = -new_w.unsqueeze(2) * new_w.unsqueeze(1)

        grad_input = torch.sum(jacobian_off_diag * grad_output.view(batch_size, 1, num_weights), dim=2)
        grad_input += diag*grad_output

        return grad_input


"""
-------------------------------------------------------------------
WEIGHT FORCED TO BE EQUAL <--- NOT WORKING

This function forces weights to be equal
along columns, this corresponds to 

w1*f1 + w2*f2 + w3*f3 = w1(f1 + f2 + f3)

as w1=w2=w3
"""

#WARNING: DOES NOT WORK

class equalWeightFunction(torch.autograd.Function):
    @staticmethod
    def foward(ctx, w):
        ctx.save_for_backward(w)
        return w * w[:, 0].unsqueeze(1)
    @staticmethod
    def backward(ctx, grad_output):
        w, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_weight = grad_output * w[:,0].unsqueeze(1)
        grad_weight = torch.sum(grad_weight, dim = 0, keepdim=True)
        grad_input[:,1:] = grad_weight.repeat(1, grad_input.size(1) - 1)
        return grad_input

class equalWeight(nn.Module):
    def forward(self, w):
        return equalWeightFunction.apply(w)








