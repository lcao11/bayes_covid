import numpy as np
import math

def mh(prior, misfit, iter, sigma):
    current = prior.init_vector()
    proposal = prior.init_vector()
    prior.sample(current)
    print("initial sample: ", current)
    accepted = np.zeros((1, prior.param_dim))
    rejected = np.zeros((1, prior.param_dim))
    accepted[0,:] = current
    rejected[0,:] = current
    current_ll = misfit.log_likelihood(current)
    current_pc = prior.log_density(current)

    for i in range(iter):
        proposal = current + np.random.normal(scale = sigma**2, size=prior.param_dim)
        proposal_ll = misfit.log_likelihood(proposal)
        proposal_pc = prior.log_density(proposal)
        log_alpha = proposal_ll - current_ll + proposal_pc - current_pc
        if math.log(np.random.rand()) < log_alpha:
            current = proposal
            accepted = np.append(accepted, np.reshape(current, (1, prior.param_dim)), axis = 0)
            current_ll = proposal_ll
            current_pc = proposal_pc
        else:
            rejected = np.append(rejected, np.reshape(proposal, (1, prior.param_dim)), axis = 0)
        
            
    return [accepted, rejected]
