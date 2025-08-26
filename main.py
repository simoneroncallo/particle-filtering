import copy as cp
import numpy as np
from tqdm import tqdm
from animation import animate
from particlefilter import likelihood, Particle

# Settings
numSteps = 100
numParticles = 400
rng = np.random.default_rng(2025)
xLimits, yLimits = [-10,10], [-10,10]  

# Arrays
objHistory = np.zeros((numSteps,2)) # Kinematics
pctsHistory = np.zeros((numSteps,numParticles,2))
estHistory = np.zeros((numSteps,2)) # Estimation 
obsHistory = np.zeros((numSteps,2)) # Observations
positions, weights = np.zeros((numParticles,2)), np.zeros(numParticles)
    
# Initialization
obj = Particle(-8,-8) # Object
init = rng.uniform(low=[-10,-10], high=[10,10], size=(numParticles,2)) # Uniform
ptcs = np.array([Particle(init[idx,0], init[idx,1], weight=1/numParticles)\
                 for idx in range(numParticles)]) # Particles

# Filtering
for time in tqdm(range(numSteps)):
    objHistory[time,:] = np.array([obj.x1, obj.x2])
    obsHistory[time] = obj.observe(rng) # Observation
    for idx in range(numParticles):  
        pctsHistory[time,idx,:] = np.array([ptcs[idx].x1, ptcs[idx].x2])
        ptcs[idx].sample(rng) # Sample
        condProb = likelihood(obsHistory[time],ptcs[idx],rng) # p(yt|xt)
        weights[idx] = ptcs[idx].weight*condProb # Update weights
    weights /= np.sum(weights) # Normalize
    for idx in range(numParticles):
        ptcs[idx].weight = weights[idx]
    
    # Estimate
    estHistory[time,0] = np.dot(pctsHistory[time,:,0],weights[:])   
    estHistory[time,1] = np.dot(pctsHistory[time,:,1],weights[:])   
    
    # Resample
    if 1/np.sum(np.square(weights)) < numParticles/4: # Threshold
        choices = rng.choice(numParticles, size = numParticles, p = weights)
        ptcsNew = np.array([Particle(ptcs[choice].x1, ptcs[choice].x2,\
                            weight=1/numParticles) for choice in choices]) 
        ptcs = cp.deepcopy(ptcsNew)
    obj.move(rng)
    
# Visualize
animate(numSteps,objHistory,pctsHistory,obsHistory,estHistory,xLimits,yLimits)
