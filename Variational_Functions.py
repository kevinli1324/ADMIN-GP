# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:12:07 2022

@author: kevin
"""

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


# generate samples from matrix normal distribution with mean A, Columnwise variance 
#V, row wise variace U
def matrix_norm(V, U, A, n = 10):
       
    mu = A.T.flatten()
    prec = torch.kron(torch.inverse(V), torch.inverse(U))
    MV1 = torch.distributions.multivariate_normal.MultivariateNormal(mu, precision_matrix= prec)
    samp = MV1.rsample((n,))
    samp = samp.reshape((n, A.shape[1], A.shape[0])).transpose(-2, -1)
    
    return samp

'''
def matrix_norm(V, U, A, n = 10):
       
    mu = A.T.flatten()
    prec = torch.kron(torch.inverse(V), torch.inverse(U))

    Vc = to_linear_operator(V).add_jitter(1e-6).cholesky().to_dense()
    Uc = to_linear_operator(U).add_jitter(1e-6).cholesky().to_dense()
    base_samp = torch.randn((n, A.shape[0], A.shape[1]))
    
    #MV1 = torch.distributions.multivariate_normal.MultivariateNormal(mu, precision_matrix= prec)
    #samp = MV1.rsample((n,))
    #samp = samp.reshape((n, A.shape[1], A.shape[0])).transpose(-2, -1)
    return A[None].double()  +   Uc[None].double()  @ base_samp.cuda().double() @ Vc.T[None].double()
'''
# get covariance matrix
def get_covxy(X, Y, theta, nu = 1):
    return (torch.div(torch.norm(X[:, None] - Y, dim=2, p=2).square(), -2*theta)).exp()*nu

def get_cov(X, theta, nu):
    return nu*((X[:, None] - X).square().sum(-1).div(-2*theta).exp())

#calculate neccesary bounds 
# calculates phi0
def fphi0(nu, N):
    return nu*N

#calcaultes phi1
def sfphi1(X, Z, theta, nu, w, A):

    ti = 1/theta

    #matrix of zjzj^T
    ztz = torch.bmm(Z[:, None, :], Z[:, :, None])[:, 0, 0]


    wi = 1/w
    xtVx = torch.matmul(X[:, None]*w[None, None, :], X[:, :, None])[:, 0, 0]
    detNvi = -(torch.log(wi).sum()+ torch.log(1 + ti*xtVx))

    ztAx = torch.matmul(torch.matmul(Z[:, None], A[None]), X.T[None])[:, 0, :]

    #left over 
    ViAtA =  torch.trace(wi[None, :]*A.T.mm(A))
    left = -.5*(ti*ztz +  ViAtA)
    cs1 = (ti**2)*(xtVx[None, :]*ztz[:, None])*(1 - ti*xtVx[None, :]/(1 + ti*xtVx[None, :]))
    cs2 = 2*ztAx*ti*( 1 - ti*xtVx/(1 + ti*xtVx))
    cs3 = ViAtA - ti*torch.matmul(X[:, None, :], torch.matmul(A.T.mm(A)[None], X[:, :, None]))[:, 0, 0]/(1 + ti*xtVx[None, :])
    csf = .5*(cs1 + cs2 + cs3)

    det_ratio = .5*A.shape[0]*(detNvi - torch.log(w).sum())
    final = nu*torch.exp(csf + det_ratio[None] + left[:, None])
    
    return final

#calcaultes phi2
def sfphi2(X, Z, theta, nu, w, A):
    ti = 1/theta
    wi = 1/w
    zij = (Z[None, :] + Z[:, None])
    Ztemp = torch.matmul(Z[:, None, :], Z[:, :, None]).flatten()
    zjzi = Ztemp[None, :] + Ztemp[:, None]
    xtVx = torch.matmul(X[:, None, :]*w[None, None], X[:, :, None]).squeeze()
    ViAtA =  torch.trace(wi[None, :]*A.T.mm(A))
    detNvi = -(torch.log(wi).sum()+ torch.log(1 + 2*ti*xtVx))

    # compute all the neccesary values

    left = -.5*(ViAtA + ti*zjzi)
    cs1 = (ti**2)*xtVx[None, None, :]*torch.matmul(zij[:, :, None, :], zij[:, :, :, None])[:, :, 0]*(1 - 2*ti*xtVx/(1 + 2*ti*xtVx))
    cs2 = 2*ti*torch.matmul(torch.matmul(zij[:, :, None, :], A[None, None]), X.T[None, None]).squeeze()*(1 - 2*ti*xtVx/(1 + 2*ti*xtVx))
    cs3 = ViAtA - 2*ti*(torch.matmul(A[None], X[:, :, None])**2)[:, :, 0].sum(-1)/(1 + 2*ti*xtVx)
    csf = .5*(cs1 + cs2 + cs3[None, None])
    det_ratio = .5*A.shape[0]*(detNvi - torch.log(w).sum())

    final = ((nu**2)*torch.exp(csf + left[:, :, None] + det_ratio[None, None])).sum(-1)
    
    return final


#calcaultes the kl divergences between variational posterior on M and the prior
    
    
    return (t1 + t2 - vdetr - n*p)



def skl(A, w, lam):
    norm_mean = A.flatten()
    norm_scale = w.repeat(A.shape[0]).sqrt()
    
    loc_diff = norm_mean - 0
    scale_ratio = norm_scale / lam
    loc_diff_scale_ratio = loc_diff / norm_scale
    t1 = torch.log(scale_ratio)
    t2 = math.sqrt(2 / math.pi) * norm_scale * torch.exp(-0.5 * loc_diff_scale_ratio.pow(2))
    t3 = loc_diff * torch.erf(math.sqrt(0.5) * loc_diff_scale_ratio)
    kl = -t1 + (t2 + t3) / lam - (0.5 * (1 + math.log(0.5 * math.pi)))
    return kl.sum(0)



def mFdq1(Y, sig, Kmm, phi0, phi1, phi2, phi2_trace):
    
    N = Y.shape[0]
    V = torch.square(1/sig)*phi2 + Kmm
    VC = CholLinearOperator(TriangularLinearOperator(to_linear_operator(V).add_jitter(1e-6).cholesky()))
    Kmmc = CholLinearOperator(TriangularLinearOperator(to_linear_operator(Kmm).add_jitter(1e-6).cholesky()))
    #Vi = torch.inverse(V.double()).float()
    #Kmi = torch.inverse(Kmm.double()).float()
    W = torch.square(1/sig)*torch.eye(Y.shape[0]).cuda() - (torch.square(1/sig)**2)*phi1.T.mm(VC.solve(phi1))
    add_terms = (-torch.square(1/sig)*phi0)/2 + (torch.square(1/sig)/2)*torch.trace(Kmmc.solve(phi2_trace))
    exp_term = -.5*Y[None, :].mm(W).mm(Y[:, None])

    return (N/2)*torch.log(torch.square(1/sig)/(2*torch.pi)) + .5*(Kmmc.logdet()) - .5*(VC.logdet()) + exp_term  + add_terms


#returns lower bound
#A_list - list of embedding matices
#Z - inducing point locations
#X, Y - Training Data
#theta - lengtscale paramters (nuisance)
#sig - noise standard deviation
#lam - shrinkage parameter
#nu - scale paramter
def mvlb1(X, Y, Z, A_list, w_list, theta, nu, sig, lam):
    Kmm = get_covxy(X = Z, Y = Z, theta = theta, nu = nu)
    
    phi0 = fphi0(nu, X.shape[0])
    
    phi1_list = []
    phi2_list = []
    KL_sum = 0
    for i in range(len(A_list)):
        phi1_list.append(sfphi1(X= X, Z = Z, theta = theta, nu = nu, w = w_list[i], A = A_list[i]))
        phi2_list.append(sfphi2(X= X, Z = Z, theta = theta, nu = nu, w = w_list[i], A = A_list[i]))
        KL_sum += skl(A = A_list[i], w = w_list[i], lam = lam[i]**2)
        
    phi1_sum = torch.cat([m[None] for m in phi1_list]).sum(0)
    phi2_trace = torch.cat([m[None] for m in phi2_list]).sum(0)
    
    phi2_sum = 0
    for i in range(len(A_list)):
        for j in range(len(A_list)):
            if i == j:
                phi2_sum += phi2_list[i]
            else:
                phi2_sum += phi1_list[i].mm(phi1_list[j].T)
    
    
    F = mFdq1(Y = Y, sig = sig, Kmm = Kmm, phi0 = phi0*len(A_list), 
             phi1 = phi1_sum, phi2 = phi2_sum, phi2_trace=  phi2_trace)
    
    #return F - .5*KL_sum
    return F - KL_sum



# functions for getting predictions 

## pred_samp_mix obtains single sample from predictive distribution of VLE-GP
## conditional on M
#XX - test design points
#X - Training design point
#Y - Training points
#Z - Inducing Pints
#M_list - list of embedding matrices
#theta -lengthscale parameter
#nu - scale paramter
#sig2 - noise variance
def pred_samp_mix(XX, X, Y, Z, M_list, theta, nu, sig2):
    Kmm = get_covxy(Z, Z, theta=  theta, nu = nu)
    #Kmi = torch.inverse(Kmm)
    Kmi = CholLinearOperator(TriangularLinearOperator(to_linear_operator(Kmm).add_jitter(1e-6).cholesky())).inverse().to_dense()
    Kmn_sum = 0
    Kxm_sum = 0
    Kxm_list = []
    for i in range(len(M_list)):
        Xn = M_list[i].mm(X.T).T
        Kmn_sum += get_covxy(Z, Xn, theta = theta, nu = nu)
        
        XXn = M_list[i].mm(XX.T).T
        Kxm_list.append(get_covxy(XXn, Z, theta = theta, nu = nu))
        Kxm_sum += Kxm_list[i]
    
    Sigma = torch.inverse((Kmm + (1/sig2)*Kmn_sum.mm(Kmn_sum.T)))
    mu = (1/sig2)*Kmm.mm(Sigma).mm(Kmn_sum).mm(Y[:, None])
    vark = Kmm.mm(Sigma).mm(Kmm)

    myp = Kxm_sum.mm(Kmi).mm(mu)[:, 0]
    
    pvar = 0
    for i in range(len(M_list)):
        Kxm = Kxm_list[i]
        #pvar += torch.diag(nu*torch.eye(XXn.shape[0]).cuda() - Kxm.mm(Kmi).mm(Kxm.T) + Kxm.mm(Kmi).mm(vark).mm(Kxm.T))
        pvar += torch.diag(nu*torch.eye(XXn.shape[0]).cuda() - Kxm.mm(Kmi).mm(Kxm.T) + Kxm.mm(Kmi).mm(vark).mm(Kmi).mm(Kxm.T))
    pvar += sig2
    
    samp = np.random.multivariate_normal(mean = myp.detach().cpu().numpy(), cov = torch.diag(pvar).cpu().detach().numpy(), size = 1)[0]
    #samp = MV.rsample((1,))
    return torch.from_numpy(samp).cuda()


# get draws from VLE-GP predictive distribution
#w_list is list of column wise standard deviations for variational posterior on M
def get_pred_mix(XX,X, Y, Z, w_list, A_list, theta, nu, sig2, n = 100):
    U = torch.eye(A_list[0].shape[0]).cuda()
    batch_list = [matrix_norm(V = torch.eye(X.shape[1]).cuda()*w_list[i]**2, U = U, A = A_list[i], n = n)[None] for i in range(len(A_list))]
    batch_array = torch.cat(batch_list, axis = 0)
    print(batch_array.shape)

    pred_store=  torch.zeros(n, XX.shape[0]).cuda()
    for i in range(n):
        pred_store[i, ] = pred_samp_mix(XX = XX, X = X, Y = Y, Z = Z, M_list = batch_array[:, i],
                                        theta = theta, nu = nu, sig2 = sig2)
    return pred_store



def get_pred_mixw(XX,X, Y, Z, w_list, A_list, theta, nu, sig2, n = 100):
    U = torch.eye(A_list[0].shape[0]).cuda()
    batch_list = [matrix_norm(V = torch.eye(X.shape[1]).cuda()*torch.diag(w_list[i])**2, U = U, A = A_list[i], n = n)[None] for i in range(len(A_list))]
    batch_array = torch.cat(batch_list, axis = 0)
    print(batch_array.shape)

    pred_store=  torch.zeros(n, XX.shape[0]).cuda()
    for i in range(n):
        pred_store[i, ] = pred_samp_mix(XX = XX, X = X, Y = Y, Z = Z, M_list = batch_array[:, i],
                                        theta = theta, nu = nu, sig2 = sig2)
    return pred_store
