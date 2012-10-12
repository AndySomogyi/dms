"""

Config Dictionary Specification
{
    struct: name of structure file (typically a PDB), required, 
    protein: name of protein section, optional.
    top: name of topology file, optional.
    top_args:    a dictionary of optional additional arguments passed to topology and pdb2gmx, these may include
                 dirname: directory where top file is generated,
                 posres: name of position restraint include file,
                 and may also include any arguments accepted by pdb2gmx, see:
                 http://manual.gromacs.org/current/online/pdb2gmx.html
                 
    md_nensemble: a optional number specifying how many 
                 
    md_args:    a dictionary of parameters used for the md run(s), these include
                nsteps: number of md steps
                multi: how many simulations to do for ensemble averaging. optional, defaults to 1
                       (I know, this is a probably not the best key name, but this is
                        the argname that mdrun takes, so used for consistency)
                
    "equilibriate_args":{"nsteps":1000},
                    
        
"""


import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='dms.log',level=logging.DEBUG)
import sys
import os

from numpy import array, zeros, transpose, dot, reshape, \
                  average, arange, sqrt, linalg, conjugate, \
                  real, correlate, newaxis, sum, mean, min, max, where
import numpy.random
import MDAnalysis

import h5py
import md
import subsystems

import gromacs.setup


class System(object):
    """
    @ivar top: name of the starting topology file.
    @ivar struct: name of the starting structure file. 
    @ivar posres: name of the current position restraint include file.
    
    @ivar ncgs: the number of coarse grained variables
    @ivar subsystems: a list of subsystems, remains constant so long 
                      as the topology does not change.
    """    
    
    # directories relative to current dir, where md will be performed.
    em_dir = "em"
    equilibriate_dir = "equlibriate"
    md_dir = "md"
    

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

        # set (or create) the topology
        self._topology(config)
            
        # load the subsystems
        # this list will remain constant as long as the topology remains constant.
        logging.info("creating subsystems")
        sslist = config["subsystems"]
        
        # python for ((car sslist) (cdr sslist))
        self.ncgs, self.subsystems = sslist[0](self, *sslist[1:])
            
            
        # solvate the system (if the config says so)
        if self.config.has_key("solvate") and self.config["solvate"]:
            # solvate automatically calls universe_changed(...)
            self.solvate()
        else:
            self.universe = MDAnalysis.Universe(self.struct)
            [s.universe_changed(self.universe) for s in self.subsystems]
                   
        
        md_nensemble = int(config["md_args"].get("multi", 1))
        md_nsteps = int(config["md_args"]["nsteps"])
        
        nrs = len(self.subsystems)
        
        # cg: nensembe x n segment x 3
        self.pos = zeros((md_nensemble,nrs,self.ncgs))
        
        # cg forces, nensembe x n segment x 3
        self.forces = zeros((md_nensemble,nrs,self.ncgs))
        
        self.velocities = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))

        logging.info("pos {}".format(self.pos.shape))
        logging.info("frc {}".format(self.forces.shape))
        logging.info("vel {}".format(self.velocities.shape))

        if nframe is not None:
            self.read_frame(config["output_file"], nframe)
            
    def _topology(self, config):
        """ 
        set up the topology 
        """
        # get the struct and top from config, generate if necessary


        # keys (and default values) from main part of config
        defs = [("struct", None), ("protein","protein"), ("top", None)]
        top_args = dict([(s[0], config.get(*s)) for s in defs])
        top_args.update(config.get("top_args", {}))
        
        top =  md.topology(**top_args)
        self.struct = top["struct"] 
        self.top = top["top"]
        self.posres = top["posres"]
        del top
            

        
        
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
        """
        do some stuff
        """
        
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
        given the current state of the system, perform a set of md runs, 
        and 
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
        
    
    
    def thermalize(self):
        p
    
    def solvate(self):
        
        if self.config.has_key("solvate") and self.config["solvate"]:
            logging.info("performing solvation")
        
            conf = self.config.copy()
            map(lambda x: conf.pop(x,None), ["struct", "top", "posres"])
        
            sol = md.solvate(self.struct, self.top, **conf)
            logging.info("completed md.solvate: {}".format(sol))
            self.struct = sol["struct"]
            self.mainselection = sol["mainselection"]
            self.ndx = sol["ndx"]
            
            # performed a solvation, so need to update the universe
            self.universe = MDAnalysis.Universe(self.struct)
            [s.universe_changed(self.universe) for s in self.subsystems]
        
            #{'qtot': qtot,
            #'struct': realpath(dirname, 'ionized.gro'),
            #'ndx': realpath(dirname, ndx),      # not sure why this is propagated-is it used?
            #'mainselection': mainselection,
            #}
            
    def minimize(self):
        conf = self.config.copy()
        map(lambda x: conf.pop(x,None), ["struct", "top", "posres"])
        mn = md.minimize(struct=self.struct, top=self.top)
        
        #{'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb'}
        
        self.struct = mn["struct"]
        print(mn)
        
        self.universe.load_new(self.struct)
        [s.minimized() for s in self.subsystems]
            

    def _write_tmp_struct(self):
        w = MDAnalysis.Writer("tmp.pdb")
        w.write(self.universe)
        w.close()
        
    def read_frame(self, hdf, nframe):
        f = h5py.File(hdf, "r")
        grp = f[str(nframe)]
        
        self.cg = array(grp["CG"],'f')
        self.forces = array(grp["FORCES"],'f')
        self.velocities = array(grp["VELOCITIES"],'f')
        self.universe.atoms.positions = array(grp["FINAL_POSITIONS"],'f')
        
        
    def test(self):
        pass

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
    "md_ensembles" : 1,
    'struct': "1OMB.pdb", 
    "subsystems" : [subsystems.RigidSubsystemFactory, "foo", "bar"],
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "md_args":{"nsteps":1000},
    "equilibriate_args":{"nsteps":1000},
    "solvate":True
    } 

def main():
    print(os.getcwd())
    
    s=System(conf)
    
    s.minimize()
    
    s.thermalize()
    
    return
    
    a={'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb', 'maxwarn':-1, "nsteps":1000}
    
    #b = md.md_defaults.copy()
    
    #b.update(a)
    
    r=md.MD_restrained("MD_POSRES", **a)
    
    print(r)
    

    #b = {'fourierspacing': 0.16, 'DispCorr': 'EnerPres', 'gen_vel': 'yes', 'integrator': 'md', 'gen_temp': 300, 'nstvout': 100, 'nstlog': 100, 'nstenergy': 100, 'ref_t': [300, 300], 'qscript': ['./local.sh'], 'maxwarn': -1, 'struct': '/home/andy/tmp/1OMB/MD_POSRES/md.gro', 'nstxtcout': '5000', 'top': '/home/andy/tmp/1OMB/top/system.top', 'gen_seed': -1, 'pcoupl': 'no', 'tau_t': [0.1, 0.1], 'constraints': 'all-bonds', 'deffnm': 'md', 'nsteps': 50000, 'tcoupl': 'Berendsen', 'rlist': 0.9, 'tc-grps': 'Protein SOL', 'lincs_order': 4, 'pme_order': 4, 'coulombtype': 'PME', 'nstlist': 5, 'nstxout': 100, 'lincs_iter': 1, 'ndx': '/home/andy/tmp/1OMB/MD_POSRES/md.ndx', 'mainselection': '"Protein"', 'constraint_algorithm': 'lincs', 'pbc': 'xyz', 'rcoulomb': 0.9, 'ns_type': 'grid', 'nstfout': '0', 'rvdw': 1.4}
    
    run = md.MDrunnerLocal("MD_POSRES", **r)
    
    r = run.run()

    
    print(r)
   
    
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


        


    

