'''
Created on Nov 2, 2012

@author: andy
'''

from numpy import array, ndarray, arange, zeros, convolve, sum, std, mean, cumsum
from random import randrange
import h5py #@UnresolvedImport
import pylab as p

def correlation(x1, x2, window=None):
    """
    Calculates the autocorrelation between two series x1 and x2.
    The series are broken up into a set of blocks of length window, 
    the correlation is the average of the convolution of each block
    with itself and the next block.
    """
    if type(x1) is not ndarray or type(x2) is not ndarray or len(x1) != len(x2):
        raise TypeError("x1 and x2 must both be ndarrays of the same length")
    
    if window is None:
        window = len(x1) / 2
        
    nblocks = int(len(x2) / window) - 1 
    corr = zeros(window)
    
    for i in arange(nblocks):
        b1 = x1[i*window:i*window + window]
        # reverse b1 for convolution
        b1 = b1[::-1]
        b2 = x2[i*window:i*window + 2*window - 1]
        #p.plot(b1)
        #p.plot(b2)
        conv = convolve(b1,b2,'valid') / float(window)
        #p.plot(conv)
        corr += conv

    return corr / nblocks

def vacf(v1, v2, acflen=None):
    """
    """
    if type(v1) is not ndarray or type(v2) is not ndarray or v1.shape != v2.shape:
        raise TypeError("x1 and x2 must both be ndarrays of the same length")
    
    shape = v1.shape
    
    if acflen is None:
        acflen = shape[1]/2 + 1
        
    corr = zeros(acflen)
    
    for i in arange(acflen):
        n = shape[0] - acflen + 1 - i
        c = 0.0
        for j in arange(n):
            a=v1[j:j+acflen,:]
            b=v2[i+j:i+j+acflen,:]
            c += ((a*b).sum(axis=1)).sum()
        print((i,n))
        corr[i]=c/n
        
    return corr / acflen

def msd(x1,x2,npts,tau):
    """ 
    calculates the mean squared displacement of a set of time series.

    @param tau: a sequence of time offsets.
    """
    #import pdb
    #pdb.set_trace()
    n = x1.shape[0]

    if npts + tau >= n:
        raise Exception("bad")

    start = randrange(0,n-npts-tau)
    result = zeros((tau,x1.shape[1]))

    for t in arange(tau):
        xx1 = x1[start  :start+npts,  :]
        xx2 = x2[start+t:start+npts+t,:]
        xx = xx1-xx2
        xx = xx**2
        result[t,:] = mean(xx,axis=0)
    return result

#def dict_msd(d,tau):
    
        

#def test_xacf(x,tau):
#    
#    xa=zeros((len(tau),x.shape[0]*3))
#    for i in arange(x.shape[0]):
#        for j in tau:
#            xx=x[i,0,:,:]
#            xa[j-1,i:i+3] = xacf(xx,xx,j)/j
#    p.plot(xa)
#    p.show()

def vacf1(v1, v2, acflen=None):
    """
    """
    if type(v1) is not ndarray or type(v2) is not ndarray or v1.shape != v2.shape:
        raise TypeError("x1 and x2 must both be ndarrays of the same length")
    
    shape = v1.shape
    
    if acflen is None:
        acflen = shape[1]/2 + 1
        
    corr = zeros(acflen)
    
    for i in arange(acflen):
        a=v1[:acflen,:]
        b=v2[i:i+acflen,:]
        corr[i]=((a*b).sum(axis=1)).sum()

    return corr / acflen
        
        

def corr(fname, blocks, what="/0/VELOCITIES", index=0):
    f=h5py.File(fname, 'r')
    vel = array(f[what])
    s=vel.shape
    window = s[2]/blocks
    
    corr = zeros((window,3))

    v1=vel[index,0,:,:]
    vx = v1[:,0]
    vy = v1[:,1]
    vz = v1[:,2]
    corr[:,0]=correlation(vx,vx, window)
    corr[:,1]=correlation(vy,vy, window)
    corr[:,2]=correlation(vz,vz, window)
    
    return corr
    

def test(fname, blocks, what="/0/VELOCITIES", index=0):
    f=h5py.File(fname, 'r')
    vel = array(f[what])
    s=vel.shape
    window = s[2]/blocks
    corr = zeros(window)
    for i in arange(s[0]):
        v1=vel[i,0,:,index]
        cc=correlation(v1,v1, window)
        corr += cc
        p.plot(cc)
        
    corr /= float(s[0])
    print(corr[0])
    
    p.plot(corr, '-o')
    p.show()
    return corr

def test2(fname, acflen=None, ensemble=0, what="/0/VELOCITIES"):
    """
    @param ensemble: index of ensemble
    """
    
    # vel: (100, 1, 5000, 3)

    f=h5py.File(fname, 'r')
    vel = f[what].value
    if acflen is None:
        acflen=vel.shape[2]/2
    vel = vel[ensemble, 0, :4*acflen, :]
    #p.plot(vacf(vel,vel,acflen))
    v=vacf1(vel,vel,acflen)
    p.plot(v)
    p.plot(cumsum(v))
    p.show()

if __name__ == "__main__":

    test2("/home/andy/tmp/C60/out.hdf", 500, 1)


    

    
