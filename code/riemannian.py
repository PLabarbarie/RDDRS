#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:51:39 2021

@author: labarbarie
"""

import torch
from torch.distributions.normal import Normal
from torch import linalg

from scipy.stats import norm
from math import ceil

import numpy as np
from time import time
import datetime
from smoothing import lower_confidence_bound, count_arr

#from torch.autograd.functional import jacobian
#os.environ["GEOMSTATS_BACKEND"] = "pytorch"


from geomstats.geometry.spd_matrices import SPDMetricAffine



#def batch_jacobian(f, x):
#    f_sum = lambda x: torch.sum(f(x), axis=0) #axis=0 fait un tensor avec 3 lignes ou chaque ligne correspond à une coordonnée 
#    return jacobian(f_sum, x)

def d_Phi(x):
    return (1/(np.sqrt(2*np.pi)))*torch.exp(-(1/2)*x**2)

def inv_Phi(x,device="cuda"):
    batch_size = x.shape[0]
    m = Normal(torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device))
    return m.icdf(x.clamp_(0.001, 0.999))

def expectation_A(W,nabla_h):
    return (1/2)*(torch.matmul(W, nabla_h) + torch.matmul(nabla_h.transpose(1, 2), W.transpose(1, 2)))

def nabla_G(model,x,c,C_t):
    # Compute the riemannian gradient of G
    batch_size = x.shape[0]
    img_size = np.prod(x.shape[1:])
    samples = 20000
    mini_batch = 10000
    E_A = 0

    for _ in range(samples//mini_batch):

        W = torch.randn((batch_size, img_size, mini_batch), device="cuda")
        eps = torch.matmul(C_t, W).transpose(1, 2).reshape(batch_size*mini_batch, 1, 28, 28)
        Z = x + eps
        Z.requires_grad = True
        out = model(Z)
        #print(c) # Faire attention à c pour bien mettre des 1 la ou il le faut
        m = torch.zeros((batch_size*mini_batch, 10), device="cuda")
        m[:, c] = 1
        #print(m)
        out.backward(m)
        nabla_h = Z.grad
        nabla_h = nabla_h.reshape(batch_size,mini_batch,img_size).float()
        E_A += expectation_A(W, nabla_h)
    E_A = E_A/samples

    # Z = x + eps
    # Z.requires_grad = True
    # out = model(Z)
    # #print(c) # Faire attention à c pour bien mettre des 1 la ou il le faut
    # m = torch.zeros((batch_size*samples,10),device="cuda")
    # m[:,c] = 1
    # #print(m)
    # out.backward(m)
    # nabla_h = Z.grad
    # nabla_h = nabla_h.reshape(batch_size,samples,img_size).float()

    #nabla_h = batch_jacobian(net,x + eps)[c]
    #nabla_h = nabla_h.reshape(batch_size,batch_size,samples,img_size)[[np.arange(batch_size)],[np.arange(batch_size)]][0]
    #nabla_h = nabla_h.view(batch_size,1,-1).float()
    #print(nabla_h.size())
    
    return ( torch.matmul(torch.matmul(C_t,E_A),C_t).detach()  )

def K(C_t):
    return (torch.linalg.eigh(C_t).eigenvalues.min())

def nabla_K(C_t):
    C_t = C_t.detach()
    C_t.requires_grad = True
    out = K(C_t)
    out.backward()
    return ( torch.matmul(torch.matmul(C_t, (C_t.grad + C_t.grad.transpose(1,2))/2 ), C_t).detach()  )

def P(C_t):
    return (torch.prod(linalg.eigh(C_t).eigenvalues**(1/784)))

def nabla_P(C_t):
    C_t = C_t.detach()
    C_t.requires_grad = True
    out = P(C_t)
    out.backward()
    return ( torch.matmul(torch.matmul(C_t, (C_t.grad + C_t.grad.transpose(1,2))/2 ), C_t).detach()  )
    

def optimize_riemannian(
        model: torch.nn.Module, batch: torch.Tensor,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, device: str = "cuda:0" ) -> torch.Tensor:
    """Optimize batch using riemannian optimization, assuming DDRS initialization point.
    Args:
        model (torch.nn.Module): trained network
        batch (torch.Tensor): inputs to certify around
        isotropic_theta (torch.Tensor): initialization anisotropic value per
            input in batch
        iterations (int): number of iterations to run the optimization
        samples (int): number of samples per input and iteration
        kappa (float): relaxation hyperparameter
        device (str, optional): device on which to perform the computations
    Returns:
        torch.Tensor: optimized anisotropic and non-diagonal covariance matrix
    """

    batch_size = batch.shape[0]   #Nombre d'entrées dans le batch
    img_size = np.prod(batch.shape[1:]) # Nombre de variables
    SPD = SPDMetricAffine(int(img_size))
    kappa = 0.00000001
    lr = 1
    # alpha = 0.55
    C_t = torch.diag_embed(isotropic_theta.clone().repeat(img_size,1).transpose(0,1))


    for gamma_t in range(2,iterations):
        # Reparameterization trick
        eps = torch.matmul(C_t, torch.randn((batch_size,img_size,samples),device="cuda")) #Reparamitrization trick

        out = model( batch + eps.transpose(1,2).reshape(batch_size*samples,1,28,28) ).reshape(batch_size, samples, -1).mean(1)#This is \psi in the algorithm

        vals, indices = torch.topk(out, 2)
        indices.transpose_(0,1)
        vals.transpose_(0, 1)

        R = inv_Phi(vals[0]) - inv_Phi(vals[1])


        nabla_R = (   torch.mul(nabla_G(model,batch,indices[0],C_t),(1/d_Phi(inv_Phi(vals[0]))).view(-1,1,1))  -
                  torch.mul(nabla_G(model,batch,indices[1],C_t),(1/d_Phi(inv_Phi(vals[1]))).view(-1,1,1))      )

        nabla_R = ( torch.mul(nabla_R, P(C_t) ) + torch.mul(nabla_P(C_t), R)
                    +
                    kappa *  ( torch.mul(nabla_R, K(C_t)) + #max(0,isotropic_theta - K(C_t)) *
                              torch.mul(nabla_K(C_t), R)
                            )
                  )

        #C_t = SPD.exp((1/(gamma_t**alpha)) * nabla_R, C_t)
        try :
            C_t = SPD.exp(lr*nabla_R, C_t)
        except AttributeError:
            return C_t , P(C_t)
        # if linalg.matrix_norm(nabla_R)<100:
        #     print(linalg.matrix_norm(nabla_R))
        #     print(C_t)
        #     C_t = SPD.exp(0.75*nabla_R, C_t)
        # else :
        #     return C_t , P(C_t)
        ###Superset condition###
        D = torch.linalg.eigh(C_t)
        C_t = torch.matmul(torch.matmul(D.eigenvectors, torch.diag_embed(torch.max(D.eigenvalues, isotropic_theta.view(-1,1)))), D.eigenvectors.transpose(1,2))

        C_t = C_t.detach()
        lr = lr * 1

    return C_t , P(C_t)


def sample_noise_riemannian(model, x, sigma_x, num, batch_size):
    #net.eval()
    with torch.no_grad():
        counts = np.zeros(10, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.matmul(sigma_x,torch.randn((1,28*28,this_batch_size), device="cuda"))
            noise = noise.transpose(1,2).reshape(this_batch_size,1,28,28)

            predictions = model(batch + noise).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), 10)
    return counts

def certify_riemannian(model, x, sigma_x , n0, n, alpha, batch_size) -> (int, float):
        # draw samples of f(x+ epsilon)
        counts_selection = sample_noise_riemannian(model, x, sigma_x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = sample_noise_riemannian(model, x, sigma_x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return -1, 0.0
        else:
            radius = norm.ppf(pABar)
            return cAHat, radius

def certify_dataset_RIEMANNIAN(loader: torch.utils.data.DataLoader, model: torch.nn.Module, N0: int, N: int,
                          sigma_iso: torch.tensor, alpha: float, batch_size: int, device: str = "cuda"):

    ff = 'multiplelr6test.csv'
    print("save into the file : ", ff)
    f = open(ff, 'w') #nom du fichier de prédiction des rayons .csv
    print("idx\tlabel\tpredict\tradius\tproxyradius\tcorrect\tsigmamin\ttimeel_elapsed", file=f, flush=True)
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    # Sigma_x = torch.zeros(1,28,28)
    # proxy_product = torch.zeros(1)

    for i, data in enumerate(loader):
        images, labels = data[0].to(device), data[1].to(device)
        for j,x in enumerate(images):
            # if i*100+j <= 3:
            #     continue
            before_time = time()
            Sigma_x , proxy_product = optimize_riemannian(model = model, batch=x.detach(),
                                                          isotropic_theta = sigma_iso[i*100+j:i*100+j+1].detach(),
                                                          iterations=100, samples=100, device="cuda")
            #print(Sigma_x)
            prediction, radius = certify_riemannian(model, x, Sigma_x[0], N0, N, alpha, batch_size)
            proxy_radius = radius * proxy_product
            correct = int(prediction == labels[j])
            sigma_min = K(Sigma_x)
            radius = radius * sigma_min
            after_time = time()
            time_elapsed = str(datetime.timedelta(
                    seconds=(after_time - before_time)))

            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{}".format(
                  i*100+j, labels[j], prediction, radius, proxy_radius.item(), correct, sigma_min.item(), time_elapsed), file=f, flush=True)
    #if i==9:
    #  break
    f.close()




    