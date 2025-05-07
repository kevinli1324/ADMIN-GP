import torch 
import numpy as np
import GPy as gp
import matplotlib.pyplot as plt
from torch.linalg import slogdet
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.distributions.multivariate_normal import MultivariateNormal as MV
from torch.distributions.laplace import Laplace as laplace
from torch.distributions.kl import kl_divergence as tkl
from torch.distributions.normal import Normal as normal
import math
from linear_operator.utils.cholesky import psd_safe_cholesky
from linear_operator.operators import CholLinearOperator, TriangularLinearOperator
from linear_operator import LinearOperator, operators, to_linear_operator
from scipy.stats import qmc as qmc


import pickle
def ht1(X):
    return torch.sin(1*X[:, 0]) + torch.cos(.5*X[:, 1])


def ht2(X):
    return torch.sin(.5*X[:, 0]*X[:, 1]) 


def ht3(X):
    return X[:, 1]*torch.cos(.75*X[:, 0])

    
def draw_exp(ntrain = 500, ntest = 300, D = 20, returnM = False, stride = 6):
    n = ntrain + ntest
    d = 2
    sampler = qmc.LatinHypercube(d=D)
    sample = sampler.random(n=n)
    X = torch.from_numpy(sample).float().cuda()
    X = X - .5
    M1 = torch.normal(0, 1, size = (d, D)).cuda()
    M1[:, :stride] = torch.normal(3, 1, size = (d, stride)).cuda()
    nX1 = M1.mm(X.T).T
    nX1 = nX1/nX1.flatten().std()

    M2 = torch.normal(0, 1, size = (d, D)).cuda()
    M2[:, (stride):(2*stride)] = torch.normal(3, 1, size = (d, stride)).cuda()
    nX2 = M2.mm(X.T).T
    nX2 = nX2/nX2.flatten().std()
    
    M3 = torch.normal(0, 1, size = (d, D)).cuda()
    M3[:, (2*stride):(3*stride)] = torch.normal(3, 1, size = (d, stride)).cuda()
    nX3 = M3.mm(X.T).T
    nX3 = nX3/nX3.flatten().std()

    Y = .4*ht1(nX1) + .3*ht2(nX2) + .3*ht3(nX3)
    Y += (torch.normal(0, 1, size = (n,)).cuda()*.15*Y.std(0)).float().cuda()

    X, Xtest = X[:ntrain], X[ntrain:]
    Y, Ytest = Y[:ntrain], Y[ntrain:]
    
    
    xmean, xs = X.mean(0), X.std(0)
    ymean, ys = Y.mean(0), Y.std(0)
    X = (X-xmean)/xs
    Y = (Y -ymean)/ys
    Xtest = (Xtest - xmean)/xs
    Ytest = (Ytest - ymean)/ys
    
    if returnM:
        return X, Y, Xtest, Ytest, [M1/nx1sd/np.sqrt(2), M2/nx2sd/np.sqrt(2), M3/nx3sd/np.sqrt(2)]
    else:
        return X, Y, Xtest, Ytest




    
