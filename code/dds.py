#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:50:56 2021

@author: labarbarie
"""


import torch
from torch.autograd import Variable
from torch.distributions.normal import Normal


from time import time
import datetime
from smoothing import certify

def OptimzeSigma(model, batch, alpha, sig_0, K, n):
    device='cuda:0'
    batch_size = batch.shape[0]

    sig = Variable(sig_0, requires_grad=True).view(batch_size, 1, 1, 1)
    m = Normal(torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device))

    for param in model.parameters():
        param.requires_grad_(False)

    #Reshaping so for n > 1
    new_shape = [batch_size * n]
    new_shape.extend(batch[0].shape)
    new_batch = batch.repeat((1,n, 1, 1)).view(new_shape)
    for _ in range(K):
        sigma_repeated = sig.repeat((1, n, 1, 1)).view(-1,1,1,1)
        eps = torch.randn_like(new_batch)*sigma_repeated #Reparamitrization trick
        out = model(new_batch + eps).reshape(batch_size, n, -1).mean(1)#This is \psi in the algorithm
        
        vals, _ = torch.topk(out, 2)
        vals.transpose_(0, 1)
        gap = m.icdf(vals[0].clamp_(0.02, 0.98)) - m.icdf(vals[1].clamp_(0.02, 0.98))
        radius = sig.reshape(-1)/2 * gap  # The radius formula
        grad = torch.autograd.grad(radius.sum(), sig)

        sig.data += alpha*grad[0]  # Gradient Ascent step

    #For training purposes after getting the sigma
    for param in model.parameters():
        param.requires_grad_(True)    

    return sig.reshape(-1)




def certify_dataset_DDS(loader: torch.utils.data.DataLoader, model: torch.nn.Module, N0: int, N: int,
                          sigma: float, alpha: float, batch_size: int, device: str = "cuda"):   



    sigma_iso = torch.zeros(10000,device="cuda") + sigma
    #sigma_iso = sigma
    for ite in range(15):
        print(ite)
        #f = open('certify_dd_MNIST_{}.csv'.format(ite), 'w')  # nom du fichier de prédiction des rayons .csv
        # print("idx\tlabel\tpredict\tradius\tproxyradius\tcorrect\tsigmamin\ttimeel_elapsed", file=f, flush=True)

        f = open('test_proba_dd_{}.csv'.format(ite), 'w')  # nom du fichier de prédiction des rayons .csv
        print("idx\tlabel\tpredict\tradius\tproxyradius\tcorrect\tsigmamin\tphi\ttimeel_elapsed", file=f, flush=True)
        for i, data in enumerate(loader):
            images, labels = data[0].to(device), data[1].to(device)
            sigma_iso[i*100:(i+1)*100] = OptimzeSigma(model, images, alpha = 0.0001, sig_0 = sigma_iso[i*100:(i+1)*100], K = 100, n = 100)

            for j,x in enumerate(images):

                before_time = time()
                prediction, radius, phi = certify(model, x, sigma_iso[i*100+j], N0, N, alpha, batch_size)
                correct = int(prediction == labels[j])
                sigma_min = sigma_iso[i*100+j].item()
                proxy_radius = radius
                after_time = time()
                time_elapsed = str(datetime.timedelta(
                        seconds=(after_time - before_time)))

                # print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{}".format(
                #       i*100+j, labels[j], prediction, radius, proxy_radius, correct, sigma_min, time_elapsed), file=f, flush=True)
                print("{}\t{}\t{}\t{:.3}\t{:.3}\t{}\t{:.3}\t{:.3}\t{}".format(
                      i*100+j, labels[j], prediction, radius, proxy_radius, correct, sigma_min, phi, time_elapsed), file=f, flush=True)
            #if i==1:
            break
        f.close()
    return sigma_iso


    