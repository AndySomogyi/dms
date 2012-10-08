import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='dms.log',level=logging.DEBUG)

from numpy import array, zeros, transpose, dot, reshape, \
                  average, arange, sqrt, linalg, conjugate, \
                  real, correlate, newaxis, sum, mean, min, max, where
import numpy.random
import MDAnalysis

import h5py
#import md
import subsystems


class System(object):
    """
    @ivar top: name of the starting topology file.
    @ivar struct: name of the starting structure file. 
    @ivar posres: name of the current position restraint include file.
    """    

    def __init__(self, config, nframe=None):
        
        """
        a list of subsystems
        """
        self.subsystems = None
        self.config = config
        
        self.timestep = 0
        
        if self.config.has_key("output_file") and nframe is None:
            self.output_file = h5py.File(self.config["output_file"], "w")
        else:
            self.output_file = None
             
        logging.debug('creating rigid subsystems from config {}'.format(config))   
    
        self.universe = MDAnalysis.Universe(config['struct'])  
        
        # load the subsystems
        sslist = config["subsystems"]
        self.ncgs, self.subsystems = sslist[0](self, *sslist[:,-1])
            
        nrs = len(self.segments)
        
        md_ensembles = int(config["md_ensembles"])
        md_frames = int(config["md_frames"])
        
        # cg: nensembe x n segment x 3
        self.pos = zeros((md_ensembles,nrs,self.ncgs))
        
        # cg forces, nensembe x n segment x 3
        self.forces = zeros((md_ensembles,nrs,self.ncgs))
        
        self.velocities = zeros((md_ensembles,nrs,md_frames,self.ncgs))

        logging.info("pos {}".format(self.pos.shape))
        logging.info("frc {}".format(self.forces.shape))
        logging.info("vel {}".format(self.velocities.shape))
        
        

        if nframe is not None:
            self.read_frame(config["output_file"], nframe)
            

        
        
    def output_file_write(self, name, value):
        """
        write a value to the output file using the given key at the current timestep.
        """
        
        if self.output_file is not None:
            keys = self.output_file.keys()
            if(keys.count(str(self.timestep))):
                grp = self.output_file[str(self.timestep)]
            else:
                grp = self.output_file.create_group(str(self.timestep))
            try:
                del grp[str(name)]
                logging.info('cleared existing output file value of \"{}/{}\"'.format(self.timestep, name))
            except KeyError:
                pass
    
            if isinstance(value, list):
                subgrp = grp.create_group(str(name))
                for i in enumerate(value):
                    subgrp[str(i[0])] = i[1]
            else:
                grp[str(name)] = value
    
            self.output_file.flush()
        
    def run(self):
        for i in range(int(self.config["cg_steps"])):
            logging.info("starting step {}".format(i))
            self.timestep = i
            self.step()
            logging.info("completed step {}".format(i))
   

    def step(self):
        # md - runs md with current state, reads in md output and populates 
        # segments statistics. 
        self.minimize()
        
        self.thermalize()
        
        self.md()
        
        self.evolve()
        
        logging.info("completed step {}".format(self.timestep))
        
        
    def evolve(self):
        beta_t = float(self.config["beta_t"])
        D = diffusion(self.velocities)
        f = mean(self.forces, axis=0).flatten()
        Df = dot(D,f)
        
        for s in enumerate(self.subsystems):
            s[1].translate(Df[s[0]:s[0]+self.ncgs])
        
        
        
    def md(self):
        """
        write out the current state, run md, and read the output.
        """
        self.pos[:] = 0.0
        self.forces[:] = 0.0
        self.velocities[:] = 0.0
        
        self.output_file_write("POSITIONS", self.universe.atoms.positions)
        
        # run md on the current state
        with md.md(self.config, self.universe) as atomics:
            for f in enumerate(atomics.files):
                print(f[1])
                self.universe.load_new(f[1])
                for ts in enumerate(self.universe.trajectory):
                    if ts[0] < self.velocities.shape[2]:
                        for s in enumerate(self.segments):
                            pos,vel,frc = s[1].frame()
                            self.pos[f[0],s[0],:] += pos
                            self.velocities[f[0],s[0],ts[0],:] = vel
                            self.forces[f[0],s[0],:] += frc
                self.universe.trajectory.close()
                
                
        # done with files, divide by n frames to get average
        self.cg[:,:,:] /= (self.velocities.shape[2])
        self.forces[:,:,:] /= (self.velocities.shape[2])  
        
        self.output_file_write("FINAL_POSITIONS", self.universe.atoms.positions)
        self.output_file_write("CG", self.cg)  
        self.output_file_write("FORCES", self.forces)  
        self.output_file_write("VELOCITIES", self.velocities)   
        
    def minimize(self):
        pass
    
    
    def thermalize(self):
        pass
        
    def read_frame(self, hdf, nframe):
        f = h5py.File(hdf, "r")
        grp = f[str(nframe)]
        
        self.cg = array(grp["CG"],'f')
        self.forces = array(grp["FORCES"],'f')
        self.velocities = array(grp["VELOCITIES"],'f')
        self.universe.atoms.positions = array(grp["FINAL_POSITIONS"],'f')

Defaults = {
            "ff":"charmm27", 
            "water":"SPC",
            "ignh":True
            }

        
from numpy.fft import fft, ifft, fftshift

def diffusion(vel, D):
    """
    calculate the matrix of diffusion coeficients.
    @param vel: a #ensembles x #subsytems x #frames x #cg array
    @return: a (#subsytems x #cg) x (#subsytems x #cg) array
    """ 
    print("shape: {}".format(D.shape))
    ndiff = int((3*len(self.segments)) ** 2)/2
    idiff = 0
    stride = ndiff / 20
    for i in arange(len(self.segments)):
        vi = self.segments[i].cg_velocities
        for j in arange(i, len(self.segments)):
            vj = self.segments[j].cg_velocities
            for ii in arange(3):
                for jj in arange(ii,3):
                    #corr = correlate(vi[:,ii], vj[:,jj])
                    #corr = diffusion.d4(vi[:,ii], vj[:,jj], 1)
                    
                    #raw_input("wait")
                    corr = sum(correlate(vi[:,ii], vj[:,jj], "full"))
                    self.D[3*i+ii,3*j+jj] = corr
                    idiff += 1
                    if idiff % stride == 0:
                        print("diffusion {}% done".format(round(100 * float(idiff) / ndiff)))
                    #print("D[[{},{}],[{},{}]]={}".format(i,ii,j,jj,corr))


def fft_corr(x1,x2):
    """FFT based autocorrelation, x is a 1 by n array. The code can be easily extended for an m by n matrix"""
    x1=reshape(x1, (1,len(x1)))
    x2=reshape(x2, (1,len(x2)))

    length = x1.shape[1]
    fft1 = fft(x1, n=(length*2-1), axis=1) #padding with zeros affects the final outcome
    fft2 = fft(x2, n=(length*2-1), axis=1) #padding with zeros affects the final outcome
    corr = ifft(fft1 * conjugate(fft2), axis=1)
    corr = real(fftshift(corr, axes=1)) #assumes no complex part, remove real for complex arrays
    corr = corr[:,length:] / range(1,length)[::-1]
    
    return corr



conf = { 
    'temperature' : 300.0, 
    'thermalise_steps' : 2000, 
    'minimise_steps' : 10000, 
    'md_frames' : 250, 
    "md_ensembles" : 1,
    'top': 'dpc_125_implicit.top', 
    'struct': "dpc_125.2.ec.pdb", 
    "md_mdp":"nvt6.mdp",
    "subsystems" : [subsystems.RigidSubsystemFactory, "foo", "bar"],
    "cg_steps":150,
    "beta_t":10.0,
    "cluster_radius":25.0
    } 

def main():
    pass
    
def hdf2trr(pdb,hdf,trr):
    u = MDAnalysis.Universe(pdb)
    w = MDAnalysis.Writer(trr,numatoms=len(u.atoms))
    f = h5py.File(hdf, "r")
    for i in range(len(f.keys())):
        frame = array(f["{}/POSITIONS".format(i)])
        u.atoms.positions = frame
        w.write(u)
    w.close()
    f.close()
    
def hdf2pdb(pdb,hdf,frame,pdb2):
    u = MDAnalysis.Universe(pdb)
    w = MDAnalysis.Writer(pdb2,numatoms=len(u.atoms))
    f = h5py.File(hdf, "r")
    
    frame = array(f["{}/POSITIONS".format(frame)])
    u.atoms.positions = frame
    w.write(u)
    w.close()
    f.close()

if __name__ == "__main__":
    main()


        


    

