import numpy as np
import math

def model_1(param, c0, t0, nt):
    
    t = np.arange(nt)+ t0
    if param[1] == 1:
        c = self.c0*np.exp(param[0]*t)
    elif param[1] > 1:
        c = np.zeros(nt)
    else:
        c = (param[0] * (1.-param[1]) * t + c0**(1.-param[1])) ** (1./(1.-param[1]))
    
    return c

def model_2(param, c0, t0, nt):
    
    t = np.arange(nt) + t0
    c = param[0] * t * np.exp(-param[1]* t **2) + c0
    
    return c
    
def model_3(param, c0, t0, nt):
    
    t = np.arange(nt) + t0
    c = param[0] * np.exp(param[1] * (1.-1./(1.-t/100.)**param[2])) + c0
    
    return c
    
def noise_var(obs_data):
    return (np.log(obs_data)*0.1+1)**2
