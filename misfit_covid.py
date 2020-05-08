import numpy as np
import math
from model_covid import model_1, model_2, model_3

class misfit_covid:
    def __init__(self, model_choice, c0, t0, obs_data, noise_variance):
    
        self.model_choice = model_choice
        self.c0 = c0
        self.t0 = t0
        self.obs_data = obs_data
        self.noise_variance = noise_variance
        
        if self.model_choice == 1 or self.model_choice == 2:
            self.param_dim = 2
        elif self.model_choice == 3:
            self.param_dim = 3
        else:
            print("model is not available")
            exit()
        
    def init_vector(self):
        return np.zeros(self.param_dim)
    
    def log_likelihood(self, x):
        
        if self.model_choice == 1:
            model_solution = model_1(x, self.c0, self.t0, self.obs_data.size)
        elif self.model_choice == 2:
            model_solution = model_2(x, self.c0, self.t0, self.obs_data.size)
        else:
            model_solution = model_3(x, self.c0, self.t0, self.obs_data.size)
            
        return - 0.5*np.linalg.norm(np.divide(model_solution - self.obs_data, self.noise_variance))
