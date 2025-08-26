# This module contains the particle filter class, as well as the state-space
# and the observation models.

import numpy as np
import scipy as sp

def stateModel(x1,x2,dx):
    """ Evolve the system in the state space. """
    if x1 < 4:
        return x1 + dx, x2 + dx
    else:
        return x1 + dx, x2 - dx/3

def observModel(x1, x2, rng):
    """ Compute the noisy observation, given the system state. """
    y1 = x1
    y2 = x2
    return np.array([y1 + rng.normal(0, 1), y2 + rng.normal(0, 1)])

def observLikelihood(yObs, pct, rng):
    """ Compute the likelihood p(yObs|x1,x2) of the actual observation
    given the particle state. """
    condPdf = sp.stats.multivariate_normal.pdf # p(y|x1,x2) 
    return condPdf(yObs, observModel(pct.x1,pct.x2,rng), [[1,0],[0,1]]) 

class Particle:
    def __init__(self,x1,x2,weight=1.0):
        self.x1 = x1 # Horizontal coordinate
        self.x2 = x2 # Vertical coordinate
        self.speed = 0.2 # Speed (for objects)
        self.weight = weight # Weight (for particles)
        
    def move(self, rng, keep = True):
        x1Upd, x2Upd = stateModel(self.x1, self.x2, self.speed)
        if keep == True:
            self.x1 = x1Upd + rng.normal(0,.1)
            self.x2 = x2Upd + rng.normal(0,.1)
            return [self.x1, self.x2]
        return [x1Upd, x2Upd]
    
    def sample(self, rng):
        mean = self.move(rng, keep = False)
        sample = rng.multivariate_normal(mean, [[.1,0],[0,.1]]) 
        self.x1, self.x2 = sample[0], sample[1]
        
    def observe(self, rng):
        return observModel(self.x1, self.x2, rng)