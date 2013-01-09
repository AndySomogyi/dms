'''
Created on Jan 6, 2013

@author: andy
'''
import numpy as n
import system

def diffusion(obj):
    """
    @param obj: An object which has cg_positions, cg_velocities and cg_forces attribites. 
                This is very convienient as both the System and the Timestep objects
                have these attributes. 
    @return: an n x n of diffusion coefficients, where n_cg is the dimensionality
    of given velocity (or force) arrays.
    
    
   
    """
    
    # 


    cg_shape = obj.cg_positions.shape
    return n.diagflat(n.ones(cg_shape[1]*cg_shape[3])*stokes(obj.temperature, r=5))

def stokes(T, r):
    """
    Estimates the diffusion cooeficient in units of Angstrom^2 / picosecond 
    for a given temperature @param T in Kelvin, and a radius @param r in Angstroms.
    
    The Einstein-Smoluchowski relation results into the Stokes-Einstein relation
    D = (KbT)/(6 pi \eta r)
    where \eta is the dynamic viscosity.
    For a laminar flow of a fluid the ratio of the shear stress to the velocity
    gradient perpendicular to the plane of shear
    
    \frac{530.516 J \text{Kb} m s T}{A \text{Kg} \text{mol} r \eta }
    
    \% \text{/.}\left\{J\to \frac{\text{Kg}*m^2}{s^2}\right\}
    
    \frac{530.516 \text{Kb} m^3 T}{A \text{mol} r s \eta }
    
    \%\text{/.}\left\{m\to 10^{10}A\right\}
    
    \frac{5.30516\times 10^{32} A^2 \text{Kb} T}{\text{mol} r s \eta }
    
    \%\text{/.}\left\{s\to 10^{12}\text{ps}\right\}
    
    \frac{5.30516\times 10^{20} A^2 \text{Kb} T}{\text{mol} \text{ps} r \eta }
    
    \%\text{/.}\left\{\text{mol}\to 6.022*10^{23}\right\}
    
    \frac{0.000880964 A^2 \text{Kb} T}{\text{ps} r \eta }
    
    \%\text{/.}\{\eta \to 0.00899\}
    
    \frac{0.0979938 A^2 \text{Kb} T}{\text{ps} r}
    
    \%\text{/.}\{\text{Kb}\text{-$>$}0.0083144621\}
    
    \frac{0.000814765 A^2 T}{\text{ps} r}
    """
    return (0.0979938 * system.KB * T ) / r
