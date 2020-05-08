import numpy as np
import math

class uniform_prior:
    def __init__(self, param_dim, upper_limit):
        self.param_dim = param_dim
        self.upper_limit = upper_limit
    
    def init_vector(self):
        return np.zeros(self.param_dim)
        
    def sample(self, s):
        r = np.random.uniform(size = self.param_dim)
        for i in range(self.param_dim):
            s[i] = r[i]*self.upper_limit[i]
            
    def within_limit(self, x):
        plausibility = True
        for i in range(self.param_dim):
            if x[i] < 0.0 or x[i] > self.upper_limit[i]:
                plausibility = False
        return plausibility
        
    def log_density(self, x):
        if self.within_limit(x):
            cost = 0.
        else:
            cost = -math.inf
        return cost
        
        
class gaussian_prior:
    def __init__(self, param_dim, mean, sigma):
        self.param_dim = param_dim
        self.mean = mean
        self.sigma = sigma

    def init_vector(self):
        return np.zeros(self.param_dim)
        
    def sample(self, s):
        r = np.random.normal(self.mean, self.sigma, size = self.param_dim)
        for i in range(self.param_dim):
            s[i] = r[i]
        
    def log_density(self, x):
        return -0.5*np.linalg.norm(np.divide(x - self.mean, self.sigma**2))
        
class kde_prior:
    def __init__(self, param_dim, kernel):
        self.param_dim = param_dim
        self.kernel = kernel

    def init_vector(self):
        return np.zeros(self.param_dim)
        
    def sample(self, s):
        r = self.kernel.resample()
        for i in range(self.param_dim):
            s[i] = r[i,0]
        
    def log_density(self, x):
        return self.kernel.logpdf(x)
