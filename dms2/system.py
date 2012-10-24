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
                 
    md:         a dictionary of parameters used for the md run(s), these include
                nsteps: number of md steps
                multi: how many simulations to do for ensemble averaging. optional, defaults to 1
                       (I know, this is a probably not the best key name, but this is
                        the argname that mdrun takes, so used for consistency)
                
    "equilibriate":{"nsteps":1000},
                    
        
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

import h5py #@UnresolvedImport
import md
import subsystems



class System(object):
    """
    @ivar top: name of the starting topology file.
    @ivar struct: name of the starting structure file. 
    @ivar posres: name of the current position restraint include file.
    
    @ivar ncgs: the number of coarse grained variables
    @ivar subsystems: a list of subsystems, remains constant so long 
                      as the topology does not change.
                      
    @ivar qtot = total charge, set by solvate
    @ivar mainselection = set by solvate, defaults to "Protein",
    @ivar ndx = set by solvate, defaults to 'main.ndx'
    """    
    
    # directories relative to current dir, where md will be performed.
    em_dir = "em"
    equilibriate_dir = "equlibriate"
    md_dir = "md"
    
    # file name of the 'system', all output will be written to this file name.
    system_struct = "system.pdb"
    

    def __init__(self, config, nframe=None):      
        """
        a list of subsystems
        """
        self.subsystems = None
        self.config = config
        
        self.timestep = 0
        
        # default values
        self.ndx = "main.ndx"
        self.mainselection = "Protein"
        
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
        # and load the initial structure
        if self.config.has_key("solvate") and self.config["solvate"]:
            # solvate automatically calls universe_changed(...)
            self.solvate()
        else:
            self.universe = MDAnalysis.Universe(self.struct)
            [s.universe_changed(self.universe) for s in self.subsystems]
                   
        
        md_nensemble = int(config.get("multi", 1))
        md_nsteps = int(config["md"]["nsteps"])
        
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
        if config.has_key("struct") and config.has_key("top"):
            logging.info("config contains a top, using {}".format(config["top"]))
            self.struct = config["struct"]
            self.top = config["top"]
            self.posres = config.get("posres", None)
        else:
            # keys (and default values) from main part of config
            logging.info("config does not contain a \"top\", autogenerating...")
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
        
        self.output_file_write("DIFFUSION", D)
        
        for s in enumerate(self.subsystems):
            s[1].translate(Df[s[0]:s[0]+self.ncgs])
            
        self.output_file_write("ATOMIC_POS", self.universe.atoms.positions)
        
        # write to struct, for next time step.
        w = MDAnalysis.Writer(self.struct)
        w.write(self.universe)
        w.close()
        

    def md(self):
        """
        take the current structure perform a set of md run(s)
        

        @precondition: 
        self.universe contains current atomic state
        self.top is valid
        
        
        @postcondition: 
        all subsystems are notified.
        """
        
        logging.debug("starting System.md()")

        conf = {"struct":System.system_struct, 
                "top":self.top, 
                "dirname":System.md_dir, 
                "deffnm":"md"}
        conf.update(self.config.get("md", {}))
        
        self._write_system_struct()
        
        logging.debug("calling md.MD with {}".format(conf))
        mdr = md.MD(**conf)
        logging.debug("md.MD returned with {}".format(mdr))
        
        #{'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb'}
        
        self.top = mdr["top"]
        self.struct = mdr["struct"]
        self.mainselection = mdr["mainselection"]
        
        # add the multi here, its an arg for mdrun, NOT grompp...
        # this is how many ensembles we do.
        # mdr["multi"] = self.pos.shape[0]
        
        # run the md
        mdr = md.run_MD(System.md_dir, **mdr)
        
        self.pos[:] = 0.0
        self.forces[:] = 0.0
        self.velocities[:] = 0.0
        
        for f in enumerate(mdr.trajectories):
            print(f[1])
            self.universe.load_new(f[1])
            for ts in enumerate(self.universe.trajectory):
                if ts[0] < self.velocities.shape[2]:
                    for s in enumerate(self.subsystems):
                        pos,vel,frc = s[1].frame()
                        self.pos[f[0],s[0],:] += pos
                        self.velocities[f[0],s[0],ts[0],:] = vel
                        self.forces[f[0],s[0],:] += frc
            self.universe.trajectory.close()
            
        # done with trajectories, reload the starting (equilibriated) structure
        # need to do this as the md result will delete the trajectory files,
        # and if these are deleted while the universe has them open, bad things
        # happen.
        self.universe.load_new(System.system_struct)
        
        # TODO
        # delete md output files...
            
        # done with files, divide by n frames to get average
        self.pos[:,:,:] /= (self.velocities.shape[2])
        self.forces[:,:,:] /= (self.velocities.shape[2])  
        
        self.output_file_write("POS", self.pos)  
        self.output_file_write("FORCES", self.forces)  
        self.output_file_write("VELOCITIES", self.velocities)   
        
        logging.debug("finished System.md()")
        

    def equilibriate(self):
        """
        take the current structure and equilibriate it via md.
        
        Loads the self.universe with the minimized structure and notifies all the 
        subsystems.
        
        @precondition: 
        self.universe contains atomic state
        
        @postcondition: 
        self.struct now refers to minimized structure generated by md
        self.universe loaded with minimized structure
        all subsystems are notified.
        """

        logging.info("starting System.equilibriate()")
        
        conf = {"struct":System.system_struct, "top":self.top, 
                "dirname":System.equilibriate_dir, "deffnm":"equilibriate"}
        conf.update(self.config.get("equilibriate", {}))
        
        self._write_system_struct()
        
        logging.debug("calling md.MD_restrained with {}".format(conf))
        mdr = md.MD_restrained(**conf)
        logging.debug("md.MD_restrained returned with {}".format(mdr))
        
        #{'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb'}
        
        self.top = mdr["top"]
        self.struct = mdr["struct"]
        self.mainselection = mdr["mainselection"]
        print(mdr)
        
        #def _run_md(self, dirname, **kwargs):
        #runner_factory = self.config.get("md_runner", md.MDrunnerLocal)
        #runner = runner_factory(dirname, **kwargs)
        #runner.run_check()
        
        md.run_MD(System.equilibriate_dir, **mdr)
        
        self.universe.load_new(self.struct)
        
        self.output_file_write("EQUILIBRIATED_POS", self.universe.atoms.positions)
        
        [s.equilibriated() for s in self.subsystems]
        
        logging.info("finished System.equilibriate()")
    
    def solvate(self):
        """
        solvate the system.
        
        currently only called on initialization
        
        TODO: implement resolvation - strip and re-generate solvent.
        
        @precondition: 
        self.struct refers to the valid structure
        
        @postcondition: 
        self.struct and self.top now refer to solvated structure generated by md, 
        both of these are over written.
        self.universe is created and initialized with the solvated structure
        all subsystems are notified.
        """
        
        if self.config.has_key("solvate") and self.config["solvate"]:
            logging.info("performing solvation")
        
            conf = self.config.copy()
            map(lambda x: conf.pop(x,None), ["struct", "top", "posres"])
        
            sol = md.solvate(self.struct, self.top, **conf)
            logging.info("completed md.solvate: {}".format(sol))
            self.qtot = sol["qtot"]
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
        else:
            logging.info("config did not specify solvation")
            
    def minimize(self):
        """
        take the current structure and minimize it via md.
        
        Loads the self.universe with the minimized structure and notifies all the 
        subsystems.
        
        @precondition: 
        self.universe contains the current atomic state
        self.top refers to a top file
        
        @postcondition: 
        self.struct now refers to minimized structure
        self.universe is loaded with the minimized structure
        all subsystems are notified.
        """
        conf = {"struct":System.system_struct, "top":self.top, 
                "ndx":self.ndx, "mainselection":self.mainselection}
        conf.update(self.config.get("minimize", {}))
        mn = md.minimize(struct=self.struct, top=self.top)
        
        # write universe to disk, md input
        self._write_system_struct()
        
        # output of minimize looks like:
        # {'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb'}
        
        self.struct = mn["struct"]
        self.mainselection = mn["mainselection"]
        self.top = mn["top"]
        print(mn)
        
        self.universe.load_new(self.struct)
        
        self.output_file_write("MINIMIZED_POS", self.universe.atoms.positions)
        [s.minimized() for s in self.subsystems]
            

    def _write_system_struct(self):
        """
        write the state of self.universe to the file system
        """
        w = MDAnalysis.Writer(System.system_struct)
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
        # {'top': '/home/andy/tmp/1OMB/top/system.top', 
        # 'dirname': 'md', 'deffnm': 'md', 'struct': '/home/andy/tmp/1OMB/equlibriate/equilibriate.gro', 
        # 'nsteps': 1000}
        self.top =  '/home/andy/tmp/1OMB/top/system.top'
        self.struct = '/home/andy/tmp/1OMB/equlibriate/equilibriate.gro'
    
        
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
    'struct': '/home/andy/tmp/1OMB/equlibriate/equilibriate.gro',
    "subsystems" : [subsystems.RigidSubsystemFactory, "protein"],
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize":{"nsteps":1000},
    "md":{"nsteps":1000},
    "multi":10,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "top":"/home/andy/tmp/1OMB/top/system.top"
    } 

C60 = { 
    'temperature' : 300.0, 
    'struct': 'C60.sol.pdb',
    "subsystems" : [subsystems.RigidSubsystemFactory, "resid 1"],
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize":{"nsteps":1000},
    "md":{"nsteps":10000},
    "multi":10,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "top":"C60.top",
    "output_file":"out.hdf"
    } 

C2 = { 
    'temperature' : 300.0, 
    'struct': 'test.pdb',
    "subsystems" : [subsystems.RigidSubsystemFactory, "protein"],
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize":{"nsteps":1000},
    "md":{"nsteps":100000},
    "multi":10,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "top":"topol.top"
    } 


def main():
    print(os.getcwd())
    
    s=System(C60)
    
    s.minimize()
    
    s.equilibriate()
    
    #s.test()
    s.md()
    

   
    
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


        


    

