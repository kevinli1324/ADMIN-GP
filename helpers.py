# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 23:42:38 2022

@author: kevin
"""
from Variational_Functions import *
import torch
import numpy as np
import GPy as gp
import properscoring as ps
import gpytorch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import RBFKernel, ScaleKernel, AdditiveStructureKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import pandas as pd
from gpytorch.models import ExactGP, AbstractVariationalGP
import rp
from rp import *
from tqdm import tqdm
#object containing the elements of the model
#A_list[i]*phi[0] constains mean of the ith embedding matrix distribution.
#Each row of Z is an inducing point
#phi[0] - lengtsale parameters to ease optimization
#phi[1] - sqrt(nu)
#phi[2] - sqrt(beta^{-1})
#w - standard deviation of columns
#lam - variance of elements in Matrix normal prior, shrinkage parameter
class Amb(torch.nn.Module):
    def __init__(self, A_list ,Z, phi, w, lam):
        super().__init__()
        self.A_list = torch.nn.ParameterList(A_list)
        self.Z = torch.nn.Parameter(Z, True)
        self.phi =torch.nn.Parameter(phi, True)
        self.w = torch.nn.Parameter(w, True)
        self.lam = torch.nn.ParameterList(lam)


# negative log likelihood for normal distribution
def nll(Y, mu, sig2):
    
    return (Y- mu)**2/(2*sig2) + .5*np.log(2*np.pi*sig2)

# optimization routine for lower bound
#X, Y - training data
#d1 - dimension of linear embedding
#J - number of additive components
#nm - number of inducing points
#tinit - intitial value of theta - generally set to 1-2
#nuinit - initial value of nu - generally set to 1-2
# verbose - print progress and loss
# iters - number of optimziation iterations
# lr - learning rate
def vim_fitb(X, Y, d1, J, nm, tinit, nuinit, verbose=  True, iters = 3000, lr = .005, lam_init = None, g = None, init = .25, rep = False, ret_loss = False):
    D = X.shape[1]
    V = torch.eye(D).cuda().double()
    A_list = [init*torch.normal(0,1, size = (d1, D)).cuda().double() for j in range(J)]
    Z = torch.normal(0, 1, size = (nm, d1)).cuda().double()

    phi = torch.tensor([tinit, nuinit, .1]).cuda().double()
    w = torch.cat([torch.ones(X.shape[1]).cuda()[None] for j in range(J)])
    W = torch.eye(X.shape[1]).cuda()
    
    
    #lam = torch.full((X.shape[1],), .01).cuda().double()
    #lam_list = [lam for i in range(J)]
    if lam_init is None:
        if d1 > 7 or D > 25:
            lam = torch.tensor([.1]).cuda().double()
        else:
            lam = torch.tensor([.2]).cuda().double()
    else:
        lam = torch.tensor([lam_init]).cuda().double()
    
    #lam_list = [torch.tensor([.1]).cuda().double() for i in range(J)]
    lam_list = [lam for i in range(J)]
    t = Amb(A_list= A_list, Z = Z, phi = phi, w = .1*torch.ones((J, D)).cuda().double(), lam = lam_list)
    epochs_iter =  tqdm(range(iters), miniters = 50)
    #geotorch.orthogonal(t, 'A_list')
    # first optimizatoin
    print("first optimization")
    optimizer = torch.optim.Adam(t.parameters(), lr= lr)
    #scheduler = MultiStepLR(optimizer, milestones=[2000], gamma=.5)
    first = True
    og = -1000
    for i in epochs_iter:
        optimizer.zero_grad()
        loss = mvlb1(X = X, Y= Y, Z = t.Z, A_list = t.A_list, w_list = t.w**2,
                     theta = t.phi[0], nu = t.phi[1]**2, 
                   sig = t.phi[-1], lam = t.lam)
        loss = -loss
        loss.backward()
        optimizer.step()
        #scheduler.step()
        #if loss < 2*X.shape[0]:
        #    optimizer = torch.optim.Adam(t.parameters(), lr= .001)
        
        #if loss < .75*X.shape[0]:
        #    break
        #if loss < X.shape[0]:
        #    optimizer = torch.optim.Adam(t.parameters(), lr= .005)
        #if i  % 20 == 0:
        #    if torch.abs(og - loss).item() < .005:
        #        break
        #    else:
        #        og = loss.item()
        
            
        if verbose:
            print(i)
            print(loss)
            print(t.phi)
            print(t.w)
            print(t.lam[0])
            print("-------------")
        if i % 50 == 0:
            epochs_iter.set_postfix(loss = loss.item())
    og = -1000
    sig = torch.tensor([t.phi[-1].clone().item()]).cuda().double()
    sig.requires_grad_(True)
    optimizer = torch.optim.Adam([sig], lr= .001)
    for i in range(250):  
        optimizer.zero_grad()
        loss = mvlb1(X = X, Y= Y, Z = t.Z, A_list = t.A_list, w_list = t.w**2, theta = t.phi[0], nu = t.phi[1]**2, 
                   sig = sig, lam = t.lam)
        loss = -loss
        loss.backward()
        optimizer.step()
        #scheduler.step()
        if i  % 20 == 0:
            if torch.abs(og - loss).item() < .001:
                break
            else:
                og = loss.item()

        if verbose:
            print(i)
            print(loss)
            print(sig)
            print(t.w)
            print(t.lam)
            print("-------------")
    
    phi = t.phi.clone()
    phi[-1] = sig.item()
    tnew = Amb(A_list= t.A_list, Z = t.Z, phi = phi, w = t.w, lam = t.lam)
    tnew.loss = loss

    if loss.item() > .7*X.shape[0] and rep:
        twhat, tloss = vim_fitb(X = X, Y = Y, d1 = d1, J = J, nm = nm, tinit = tinit, nuinit = nuinit, verbose=  verbose, iters = iters, lr = lr, lam_init = lam_init, g = g, init = init, rep = False, ret_loss = True)
        if tloss < loss:
            tnew = twhat
        
    
    if ret_loss:
        return tnew, loss.item()
    else:
        return tnew

# draw normal samples#

def draw_samps_t(mu_vec, var_vec, n = 1000):
    
    fill = torch.zeros(n, mu_vec.shape[0]).cuda()
    for i in range(n):
        fill[i, :] = torch.normal(mean = mu_vec, std = var_vec.sqrt())
    
    return fill

def draw_samps_n(mu_vec, var_vec, n = 1000):
    
    fill = np.zeros( (n, mu_vec.shape[0]))
    for i in range(n):
        fill[i, :] = np.random.normal(loc = mu_vec, scale = np.sqrt(var_vec))
    
    return fill


#monte carlo calculation of predictive negative log likelihood
def vle_nll(t, X, Y, Xtest, Ytest, w_list, samps = 100, l = True):
    if l:
        A_list = t.A_list
    else:
        A_list = [t.A]
    
    U = torch.eye(A_list[0].shape[0]).cuda()
    batch_list = [matrix_norm(V = torch.eye(X.shape[1]).cuda()*w_list[i]**2, U = U, A = A_list[i], n = samps)[None] for i in range(len(A_list))]
    batch_array = torch.cat(batch_list, axis = 0)
    nll_vec = np.zeros(samps)
    for i in range(samps):
        myp, pvar =  get_mean_pred(t, X = X, Y = Y, Xtest = Xtest, M_list = A_list)
        nll_vec[i] = ((Ytest- myp)**2/(2*pvar) + .5*torch.log(2*torch.pi*pvar)).detach().cpu().numpy().mean()
    
    return nll_vec.mean(0)

# helper function just collects statistics 
def collect_stats(Ytest, Ydraws, sig2, mu, nll_val, loss = None, train_rmse = None):
    return_dict = {}
    return_dict['rmse'] = np.sqrt(np.square(Ytest - mu).mean())
    return_dict['nll'] = nll_val
    #return_dict['rmspe'] = np.sqrt(np.square(((Ytest - mu)/Ytest).mean()))
    crps_vec  = np.zeros(Ytest.shape[0])
    for i in range(Ytest.shape[0]):
        crps_vec[i] = ps.crps_ensemble(Ytest[i], Ydraws[:, i])
    
    return_dict['crps'] = crps_vec.mean()
    
    lower = np.quantile(Ydraws, .025, axis = 0)
    upper = np.quantile(Ydraws, .975, axis = 0)
    return_dict['cov'] = np.mean(np.logical_and(Ytest < upper, Ytest > lower))
    return_dict['unc'] = Ydraws.std(0).mean() 
    return_dict['loss'] = loss
    if not train_rmse is None:
        return_dict['train_rmse'] = train_rmse
    return return_dict


# get predictions conditional on mean embedding matrix
def get_mean_pred(t, X, Y, Xtest, M_list = []):

    if len(M_list) > 0:
        M_list = M_list
    else:
        M_list = t.A_list
    theta = t.phi[0]
    nu = t.phi[1]**2
    sig2 = t.phi[-1]**2
    XX = Xtest
    Z = t.Z
    
    Kmm = get_covxy(Z, Z, theta=  theta, nu = nu)
    Kmi = torch.inverse(Kmm)
    
    Kmn_sum = 0
    Kxm_sum = 0
    Kxm_list = []
    for i in range(len(M_list)):
        Xn = M_list[i].mm(X.T).T
        Kmn_sum += get_covxy(Z, Xn, theta = theta, nu = nu)
    
        XXn = M_list[i].mm(XX.T).T
        Kxm = get_covxy(XXn, Z, theta = theta, nu = nu)
        Kxm_list.append(Kxm)
        Kxm_sum += Kxm_list[i]
        
    
    Sigma = torch.inverse((Kmm + (1/sig2)*Kmn_sum.mm(Kmn_sum.T)))
    muk = (1/sig2)*Kmm.mm(Sigma).mm(Kmn_sum).mm(Y[:, None])
    vark = Kmm.mm(Sigma).mm(Kmm)
    
    
    pvar = 0
    for i in range(len(M_list)):
        Kxm = Kxm_list[i]
        pvar += torch.diag(nu*torch.eye(XXn.shape[0]).cuda() - Kxm.mm(Kmi).mm(Kxm.T) + Kxm.mm(Kmi).mm(vark).mm(Kmi).mm(Kxm.T))
    pvar += sig2


    myp = Kxm_sum.mm(Kmi).mm(muk)[:, 0]
    
    return myp, pvar

# All functions below just run the model and retrieves performance statistics #
def vim_stats(X, Y, Xtest, Ytest, d1, J, nm, verbose = False, iters = 4000,
              lr = .0025, ft = 1.25, retry = True, tinit = 1., nuinit = None, g =  None, lam_init = None):
    X, Y, Xtest, Ytest = X.double(), Y.double(), Xtest.double(), Ytest.double()
    if nuinit is None:
        if d1*J < 10:
            nuinit = 1.
        else:
            nuinit = 2
    
    t = vim_fitb(X = X, Y = Y, d1 = d1,nm = nm
                , tinit = tinit, nuinit = nuinit, J = J,iters = iters, g = g, verbose = verbose, lr = lr, lam_init = lam_init)
        
    myp, pvar = get_mean_pred(t = t, X = X, Y= Y,Xtest = Xtest)
    
    M_list = t.A_list
    preds = get_pred_mix(XX = Xtest, X = X, Y = Y, Z = t.Z, w_list = (t.w), A_list = t.A_list, 
                        theta = t.phi[0], nu = t.phi[1]**2, sig2 = t.phi[-1]**2, n = 500)
    mpred = preds.mean(0)
    pred_std = preds.std(0)
    nllr =  vle_nll(t = t, X = X, Y=Y, 
                   Xtest = Xtest, Ytest = Ytest, 
                   w_list = t.w, samps = 100, l = True)

    return_dict = collect_stats(Ytest = Ytest.detach().cpu().numpy(), 
                                Ydraws = preds.detach().cpu().numpy(), 
                                sig2 = pred_std.square().detach().cpu().numpy(), 
                                mu = myp.detach().cpu().numpy(), 
                                nll_val = nllr, 
                                loss = t.loss)
    
    if (t.phi[1] < .05 or torch.abs(t.loss) > ft*X.shape[0]) and retry:
        print(return_dict)
        raise Exception("Optimization Failure")
    
    return return_dict


def gpy_stats(X, Y, Xtest, Ytest):
    #X, Y, Xtest, Ytest = X.float(), Y.float(), Xtest.float(), Ytest.float()
    Xc = X.cpu().numpy()
    Yc = Y.cpu().numpy()
    ker = gp.kern.RBF(input_dim=Xc.shape[1], ARD = True, lengthscale = 2) + gp.kern.White(input_dim=Xc.shape[1])
    gpy_mod = gp.models.GPRegression(Xc,Yc[:, None], ker)
    
    gpy_mod.optimize("bfgs", start = None)

    Xtc = Xtest.cpu().numpy()
    Ytc = Ytest.cpu().numpy()

    mu, var = gpy_mod.predict(Xtc)
    
    pred_samps = draw_samps_n(mu[:, 0], var[:, 0], n= 500)
    nllr = nll(Y = Ytc, mu = mu[:, 0], sig2 = var[:, 0]).mean()

    return_dict = collect_stats(Ytest= Ytc, Ydraws = pred_samps, 
                                sig2 = var[:, 0], mu = mu[:, 0], nll_val = nllr)
    
    return return_dict


def dle_stats(X, Y, Xtest, Ytest, d1, retry = False, ft = 1.25, lr = .01, train_stats = False):
    X, Y, Xtest, Ytest = X.float(), Y.float(), Xtest.float(), Ytest.float()
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood, d1):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims = None))
            self.M = torch.nn.Parameter(torch.normal(0, 1.2, size = (d1, train_x.shape[1])).cuda())
        def forward(self, x):
            xr = self.M.mm(x.T).T
            mean_x = self.mean_module(xr)
            covar_x = self.covar_module(xr)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
     
    

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(X, Y, likelihood, d1)
    model = model.cuda()

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(2000):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(X)
        # Calc loss and backprop gradients
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()
        
        
    model.eval()
    f_preds = model(Xtest)
    y_preds = likelihood(model(Xtest))

    f_mean = f_preds.mean
    f_var = f_preds.variance
    f_samps = draw_samps_t(f_mean, (f_var + model.likelihood.noise), n = 500)
    nllr = nll(Y = Ytest.cpu().numpy(), mu = f_mean.detach().cpu().numpy(),
               sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy()).mean()

    if not train_stats:
        return_dict = collect_stats(Ytest = Ytest.detach().cpu().numpy(), 
                                    Ydraws = f_samps.detach().cpu().numpy(), 
                                    sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy(), 
                                    mu = f_mean.detach().cpu().numpy(), 
                                    nll_val = nllr)
    else:
        train_rmse = (Y - model(X).mean).square().mean().sqrt().item()
        return_dict = collect_stats(Ytest = Ytest.detach().cpu().numpy(), 
                                    Ydraws = f_samps.detach().cpu().numpy(), 
                                    sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy(), 
                                    mu = f_mean.detach().cpu().numpy(), 
                                    nll_val = nllr, 
                                    train_rmse = train_rmse)

    if (loss > ft*X.shape[0] or loss < -ft*X.shape[0]) and retry:
        #print(return_dict)
        raise Exception("Optimization Failure")

    return return_dict


class ScaledProjectionKernel(gpytorch.kernels.Kernel):
    def __init__(self, projection_module, base_kernel, prescale=False, ard_d=None, learn_proj=False, **kwargs):
        self.has_lengthscale = True
        super(ScaledProjectionKernel, self).__init__(ard_d=ard_d, **kwargs)
        self.projection_module = projection_module
        self.learn_proj = learn_proj
        if not self.learn_proj:
            for param in self.projection_module.parameters():
                param.requires_grad = False

        self.base_kernel = base_kernel
        for param in self.base_kernel.parameters():
            param.requires_grad = False

        self.prescale = prescale

    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        eq = torch.equal(x1, x2)
        if self.prescale:
            x1 = x1.div(self.lengthscale)
        x1 = self.projection_module(x1)
        if not self.prescale:
            x1 = x1.div(self.lengthscale)

        if eq:
            x2 = x1
        else:
            if self.prescale:
                x2 = x2.div(self.lengthscale)
            x2 = self.projection_module(x2)
            if not self.prescale:
                x2 = x2.div(self.lengthscale)
        return self.base_kernel(x1, x2, diag=diag, last_dim_is_batch=last_dim_is_batch, **params)
    



def dpa_stats(X, Y, Xtest, Ytest, num_proj, retry = False, ft = 1.25, opt = True, lr = .01, train_stats = False):
    X, Y, Xtest, Ytest = X.float(), Y.float(), Xtest.float(), Ytest.float()
    class ExactGPModel(ExactGP):
        """Basic exact GP model with const mean and a provided kernel"""
        def __init__(self, train_x, train_y, likelihood, kernel):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = kernel

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    n, d = X.shape
    
    num_projs = num_proj
    
    # Draw random projections and store in a linear module
    # Here, we are drawing 20 Gaussian projections into 1 dimension.
    projs = [rp.gen_rp(d, 1, dist='gaussian') for _ in range(num_projs)]
    proj_module = torch.nn.Linear(d, num_projs, bias=False)
    proj_module.weight.data = torch.cat(projs, dim=1).t()
    
    # Create the additive model that operates over these projections
    # Fixing the outputscale and lengthscale of the base kernels.
    base_kernel = RBFKernel()
    base_kernel.initialize(lengthscale=torch.tensor([1.]))
    base_kernel = ScaleKernel(base_kernel)
    base_kernel.initialize(outputscale=torch.tensor([1/num_projs]))
    
    # Combine into a single module.
    kernel = ScaledProjectionKernel(proj_module, base_kernel, 
                                    prescale=True,
                                    #ard_num_dims=X.shape[1],
                                    learn_proj=opt)
    # Or, just call the method from training_routines that wraps this initialization
    # from training_routines import create_additive_rp_kernel
    # create_additive_rp_kernel(d, num_projs, learn_proj=False, kernel_type='RBF', 
    #                           space_proj=False, prescale=False, ard=True, k=1, 
    #                           proj_dist='gaussian')
    
    kernel = ScaleKernel(kernel) # Optionally wrap with an additional ScaleKernel 
    
    # Create an ExactGP model with this kernel
    likelihood = GaussianLikelihood()
    likelihood.noise = .25
    model = ExactGPModel(X, Y, likelihood, kernel)
    model = model.cuda()
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    
    mll.train()
    optimizer = torch.optim.Adam(mll.parameters(), lr=lr)
    for iteration in range(2000):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output, Y)
        loss.backward()
        optimizer.step()

    #test rmse
    model.eval()
    f_preds = model(Xtest)
    #f_preds = likelihood(model(Xtest))
    
    f_mean = f_preds.mean
    f_var = f_preds.variance
    sig2 = model.likelihood.noise

    f_samps = draw_samps_t(f_mean, (f_var + sig2), n = 500)
    
    nllr = nll(Y = Ytest.detach().cpu().numpy(), 
               mu = f_mean.detach().cpu().numpy(), 
               sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy()).mean()
    if not train_stats:
        return_dict = collect_stats(Ytest = Ytest.detach().cpu().numpy(), 
                                    Ydraws = f_samps.detach().cpu().numpy(), 
                                    sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy(), 
                                    mu = f_mean.detach().cpu().numpy(), 
                                    nll_val = nllr)
    else:
        train_rmse = (Y - model(X).mean).square().mean().sqrt().item()
        return_dict = collect_stats(Ytest = Ytest.detach().cpu().numpy(), 
                                    Ydraws = f_samps.detach().cpu().numpy(), 
                                    sig2 = (f_var + model.likelihood.noise).detach().cpu().numpy(), 
                                    mu = f_mean.detach().cpu().numpy(), 
                                    nll_val = nllr, 
                                   train_rmse = train_rmse)
    
    if (loss > ft*X.shape[0] or loss < -ft*X.shape[0]) and retry:
        #print(return_dict)
        raise Exception("Optimization Failure")

    return return_dict


#takes in lis
