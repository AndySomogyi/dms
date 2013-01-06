'''
Created on Jan 6, 2013

@author: andy
'''
import numpy as n

def diffusion(obj):
    """
    @param obj: An object which has cg_positions, cg_velocities and cg_forces attribites. 
                This is very convienient as both the System and the Timestep objects
                have these attributes. 
    @return: an n x n of diffusion coefficients, where n_cg is the dimensionality
    of given velocity (or force) arrays.
    """
    cg_shape = obj.cg_positions.shape
    return n.diagflat(n.ones(cg_shape[1]*cg_shape[3]))
