#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 10:49:49 2021

@author: labarbarie
"""
import os

import torch

import torchvision
import torchvision.transforms as transforms
import pandas as pd
os.environ["GEOMSTATS_BACKEND"] = "pytorch"



from train import CNN, train, test
from smoothing import  certify_dataset_COHEN
from dds import certify_dataset_DDS
from ancer import certify_dataset_ANCER
from riemannian import certify_dataset_RIEMANNIAN



def main():
    sigma = 0.5
    print("sigma = ",sigma)
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    batch_size = 100
 
    trainset = torchvision.datasets.MNIST(root='/scratchf/MNIST', train=True,
                                            download=True, transform=transforms.ToTensor())
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True)
    
    testset = torchvision.datasets.MNIST(root='/scratchf/MNIST', train=False,
                                           download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False)
    
    classes = ("0","1","2","3","4","5","6","7","8","9")
    
    net = CNN()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    #Paramètres
    #print("TRAINING")
    #train(loader = trainloader, model = net, epochs = 10, sigma = sigma, device = "cuda")
    #torch.save(net.state_dict(), 'LeNet5_0.25.pth')
    net.load_state_dict(torch.load('LeNet5_{}.pth'.format(sigma)))
    net.eval()
    net.to(device)
    test(loader = testloader, model = net, sigma = sigma, device = "cuda")

        
    N = 100000
    alpha = 0.001
    batch_size = 400
    N0 = 100
    os.chdir("results/{}/".format(sigma))

    # print("CERTIFICATION COHEN")
    # certify_dataset_COHEN(loader = testloader, model = net, N0 = N0, N = N,
    #                       sigma = sigma, alpha = alpha, batch_size = batch_size, device = "cuda")


    # print("CERTIFICATION DDS")
    # sigma_iso = certify_dataset_DDS(loader = testloader, model = net, N0 = N0, N = N,
    #                       sigma = sigma, alpha = alpha, batch_size = batch_size, device = "cuda")

    sigma_iso = pd.read_csv("certify_dds_MNIST.csv",sep="\t")
    sigma_iso = torch.tensor(sigma_iso["sigmamin"].values,device="cuda").float()

    print("CERTIFICATION ANCER")
    certify_dataset_ANCER(loader = testloader, model = net, N0 = N0, N = N,
                         sigma_iso = sigma_iso, alpha = alpha, batch_size = batch_size, device = "cuda")


    # print("CERTIFICATION RIEMANNIAN")
    # certify_dataset_RIEMANNIAN(loader = testloader, model = net, N0 = N0, N = N,
    #                   sigma_iso = sigma_iso, alpha = alpha, batch_size = batch_size, device = "cuda")

if __name__ == "__main__":
    
    main()


