#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:51:23 2021

@author: labarbarie
"""

from scipy.stats import norm
from math import ceil
from statsmodels.stats.proportion import proportion_confint
import torch



import numpy as np
from time import time
import datetime

def sample_noise(model, x, sigma_x, num, batch_size):
    #net.eval()
    with torch.no_grad():
        counts = np.zeros(10, dtype=int)
        for _ in range(ceil(num / batch_size)):
            this_batch_size = min(batch_size, num)
            num -= this_batch_size
            batch = x.repeat((this_batch_size, 1, 1, 1))
            noise = torch.randn_like(batch) * sigma_x
            predictions = model(batch + noise).argmax(1)
            counts += count_arr(predictions.cpu().numpy(), 10)
    return counts

def count_arr(arr, length): 
    counts = np.zeros(length, dtype=int)
    for idx in arr:
        counts[idx] += 1
    return counts

def lower_confidence_bound(NA, N, alpha):
    return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]

def certify(model, x, sigma_x , n0, n, alpha, batch_size) -> (int, float):
        # draw samples of f(x+ epsilon)
        counts_selection = sample_noise(model, x, sigma_x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = sample_noise(model, x, sigma_x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = lower_confidence_bound(nA, n, alpha)

        if pABar < 0.5:
            return -1, 0.0, 0.0
        else:
            radius = sigma_x * norm.ppf(pABar)
            return cAHat, radius, norm.ppf(pABar)
        
        
def certify_dataset_COHEN(loader: torch.utils.data.DataLoader, model: torch.nn.Module, N0: int, N: int,
                          sigma: float, alpha: float, batch_size: int, device: str = "cuda"):
        
    f = open('certify_MNIST.csv', 'w') #nom du fichier de prédiction des rayons .csv
    print("idx\tlabel\tpredict\tradius\tproxyradius\tcorrect\tsigmamin\ttimeel_elapsed", file=f, flush=True)
    
    for i, data in enumerate(loader):
        images, labels = data[0].to(device), data[1].to(device)
        for j,x in enumerate(images):
            before_time = time()
            prediction, radius = certify(model, x, sigma, N0, N, alpha, batch_size) 
            correct = int(prediction == labels[j])
            proxy_radius = radius
            after_time = time()
            time_elapsed = str(datetime.timedelta(
                    seconds=(after_time - before_time))) 
            print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{}".format(
                  i*100+j, labels[j], prediction, radius, proxy_radius, correct, sigma, time_elapsed), file=f, flush=True)
        #if i==1:
        #  break
        #break
    f.close()



    