#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:51:31 2021

@author: labarbarie
"""

import torch
from torch.autograd import Variable

import ddsmoothing
from scipy.stats import norm
from math import ceil

import numpy as np
from time import time
import datetime
from smoothing import lower_confidence_bound, count_arr

def optimize_ancer(
        model: torch.nn.Module, batch: torch.Tensor,
        certificate, learning_rate: float,
        isotropic_theta: torch.Tensor, iterations: int,
        samples: int, kappa: float, device: str = "cuda"
) -> torch.Tensor:
    """Optimize batch using ANCER, assuming isotropic initialization point.
    Args:
        model (torch.nn.Module): trained network
        batch (torch.Tensor): inputs to certify around
        certificate (Certificate): instance of desired certification object
        learning_rate (float): optimization learning rate for ANCER
        isotropic_theta (torch.Tensor): initialization isotropic value per
            input in batch
        iterations (int): number of iterations to run the optimization
        samples (int): number of samples per input and iteration
        kappa (float): relaxation hyperparameter
        device (str, optional): device on which to perform the computations
    Returns:
        torch.Tensor: optimized anisotropic thetas
    """
    batch_size = batch.shape[0]
    img_size = np.prod(batch.shape[1:])

    # define a variable, the optimizer, and the initial sigma values
    theta = Variable(isotropic_theta, requires_grad=True).to(device)
    optimizer = torch.optim.Adam([theta], lr=learning_rate)
    initial_theta = theta.detach().clone()

    # reshape vectors to have ``samples`` per input in batch
    new_shape = [batch_size * samples]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1, samples, 1, 1)).view(new_shape)

    # solve iteratively by projected gradient ascend
    for _ in range(iterations):
        theta_repeated = theta.repeat(1, samples, 1, 1).view(new_shape)
        # Reparameterization trick
        noise = certificate.sample_noise(new_batch, theta_repeated)
        out = model(
            new_batch + noise
        ).reshape(batch_size, samples, -1).mean(dim=1)

        vals, _ = torch.topk(out, 2)
        gap = certificate.compute_proxy_gap(vals)


        prod = torch.prod(
            (theta.reshape(batch_size, -1))**(1/img_size), dim=1)
        proxy_radius = prod * gap

        radius_maximizer = - (
            proxy_radius.sum() +
            kappa *
            (torch.min(theta.view(batch_size, -1), dim=1).values*gap).sum()
        )
        radius_maximizer.backward()
        optimizer.step()
        optimizer.zero_grad()
        # project to the initial theta
        with torch.no_grad():
            torch.max(theta, initial_theta, out=theta)

    return theta

def certify_ancer(model, x, sigma_x , n0, n, alpha, batch_size) -> (int, float):
        # draw samples of f(x+ epsilon)
        counts_selection = sample_noise_ANCER(model, x, sigma_x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = sample_noise_ANCER(model, x, sigma_x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = lower_confidence_bound(nA, n, alpha)
        # print(pABar)
        if pABar < 0.5:
            return -1, 0.0
        else:
            radius = norm.ppf(pABar)
            return cAHat, radius
        
def sample_noise_ANCER(model, x, sigma_x, num, batch_size):
    #net.eval()
    with torch.no_grad():
        counts = np.zeros(10, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch) * sigma_x.repeat(this_batch_size, 1, 1, 1)
            # noise = torch.mul(torch.randn((1,28*28,this_batch_size), device="cuda"),sigma_x.reshape(1,28*28,1))
            # noise = noise.transpose(1,2).reshape(this_batch_size,1,28,28)
            predictions = model(batch + noise).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), 10)
    return counts    



def certify_dataset_ANCER(loader: torch.utils.data.DataLoader, model: torch.nn.Module, N0: int, N: int,
                          sigma_iso: torch.tensor, alpha: float, batch_size: int, device: str = "cuda"):   
  

    certificate = ddsmoothing.certificate.L2Certificate(1, device = "cuda")
    sigma_ancer = torch.ones((10000,1,28,28),device="cuda") * sigma_iso.reshape(-1, 1, 1, 1)
    f = open('testANCERlr2.csv', 'w') #nom du fichier de prédiction des rayons .csv
    print("idx\tlabel\tpredict\tradius\tproxyradius\tcorrect\tsigmamin\ttimeel_elapsed", file=f, flush=True)

    for i, data in enumerate(loader):
        images, labels = data[0].to(device), data[1].to(device)
        sigma_ancer[i*100:(i+1)*100] = optimize_ancer(
                model = model, batch = images,
                certificate = certificate, learning_rate = 1,
                isotropic_theta = sigma_ancer[i*100:(i+1)*100], iterations = 100,
                samples = 100, kappa = 2, device = "cuda")
        for j,x in enumerate(images):
            before_time = time()
            prediction, radius = certify_ancer(model, x, sigma_ancer[i*100+j:i*100+j+1], N0, N, alpha, batch_size)
            proxy_radius = radius * torch.prod((sigma_ancer[i*100+j:i*100+j+1]**(1/784)).reshape(-1))
            correct = int(prediction == labels[j])
            sigma_min = sigma_ancer[i*100+j].min().item()
            radius = radius * sigma_min
            after_time = time()
            time_elapsed = str(datetime.timedelta(
                    seconds=(after_time - before_time)))

            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{}".format(
                  i*100+j, labels[j], prediction, radius, proxy_radius.item(), correct, sigma_min, time_elapsed), file=f, flush=True)
        #if i==1:
        #  break
    f.close()
    
    
    
    
    

    
    
    
    
    
    
    
    
    