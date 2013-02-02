'''
Created on Jan 6, 2013

@author: andy
'''
import numpy as np
import system
from numpy.fft import fft, ifft, fftshift

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
    return np.diagflat(n.ones(cg_shape[1]*cg_shape[3])*stokes(obj.temperature, r=5))

def diff_from_vel(v, dt):
    """
    @param src: a nframe * nsubsystem * nop * 3 array
    @return: the diffusion tensor
    """
    Nsubsys = v.shape[2]
    NCG = v.shape[3]
    Ndim = v.shape[4]

    dtensor = np.zeros([Nsubsys, NCG, Ndim, Nsubsys, NCG, Ndim], 'f')
    
    for ri in arange(dtensor.shape[0]):
        for rj in arange(dtensor.shape[1]):
            for rk in arange(dtensor.shape[2]):
                dtensor[ri,rj,rk,ri,rj,rk] = diff_from_corr(v[:,:,ri,rj,rk],v[:,:,ri,rj,rk],dt)

    size = Nsubsys * NCG * Ndim
    return np.reshape(dtensor,(size,size))

def diff_from_corr(vi, vj, dt):
    """
    calculate the diffusion coefecient for a pair of time series vi, vj
    @param vi: an n * nt array of time series
    @param vj: an n * nt array of time series
    sampled at interval dt
    """
    
    # the velocity correlation function for a time average.
    corr = np.min(np.array([vi.shape[1],vj.shape[1]]))

    for i in np.arange(vi.shape[0]):
        corr += Correlation(vi[i,:],vj[i,:])
        
    # average correlation func        
    corr /= float(vi.shape[0])      
    
    # inaccurate integration (for only 4 points)  
    # Best to use orthogonal polynomials for fitting
    # the ACs, but for now keeps this for comparison
    # with snw. 
    return np.trapz(corr[:4],dx=dt)

def Correlation(x,y):
    """
    FFT-based correlation, much faster than numpy autocorr
    x and y are row-based vectors.
    """

    lengthx = x.shape[0]
    lengthy = y.shape[0]

    x = np.reshape(x,(1,lengthx))
    y = np.reshape(y,(1,lengthy))

    fftx = fft(x, 2 * lengthx - 1, axis=1) #pad with zeros
    ffty = fft(y, 2 * lengthy - 1, axis=1)

    corr_xy = ifft(fftx * np.conjugate(ffty), axis=1)
    corr_xy = np.real(fftshift(corr_xy, axes=1)) #should be no imaginary part

    corr_yx = ifft(ffty * np.conjugate(fftx), axis=1)
    corr_yx = np.real(fftshift(corr_yx, axes=1))

    corr = 0.5 * (corr_xy[:,lengthx:] / range(1,lengthx)[::-1] + corr_yx[:,lengthy:] / range(1,lengthy)[::-1])
    return np.reshape(corr,corr.shape[1])

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
