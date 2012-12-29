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
    
    @group hdf file structure:
    All state variables of the system are saved in the hdf file, this is usefull for
    restarting or debugging. All information required for a restart is saved in the hdf 
    file. This means that ONLY the hdf file is required for a restart, nothing else. 
    
    The key names will correspond to the file name.
    
    All source files are copied into the "src_files" group. 
    
    The files used by the sytem are stored in the "files" group, however, 
    all items in this group are soft links to either one of the source files, or
    new files located in a timestep group.
    
    The "/files" group will contain at a minimum:
        struct.pdb
        topol.top
        index.ndx
        
    
    "/timesteps" is a group which contains all timesteps, this has subgroups
    names "0", "1", ... "N". 
    
    "/current_timestep" is a soft link to the timestep group currently being processed.
    This will not exist on a newly created file.
    
    "/prev_timestep" is a soft link to the previously completed timestep. This is 
    used for restarts. It will not exist on newly created files. 
    
    
        
    
        
            
    
    
    
                 
"""


import logging
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename='dms.log',level=logging.DEBUG)
import sys
import os

from numpy import array, zeros, transpose, dot, reshape, \
                  average, arange, sqrt, linalg, conjugate, \
                  real, correlate, newaxis, sum, mean, min, max, \
                  where, pi, arctan2, sin, cos, fromfile, uint8
import numpy.random
import MDAnalysis

import h5py #@UnresolvedImport
import md
import subsystems
import util


# change MDAnalysis table to read carbon correctly
import MDAnalysis.topology.tables
MDAnalysis.topology.tables.atomelements["C0"]="C"


CURRENT_TIMESTEP = "current_timestep" 
PREV_TIMESTEP = "prev_timestep"
SRC_FILES = "src_files"
FILE_DATA_LIST = ["struct.pdb", "topol.top", "index.ndx", "posres.itp"]
TIMESTEPS = "timesteps"
CONFIG = "config"
STRUCT_PDB = "struct.pdb"
TOPOL_TOP = "topol.top"
INDEX_NDX = "index.ndx"

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
    @ivar ndx = the index file, set by solvate, defaults to 'main.ndx'
    
    @ivar subsystems: a list of subsystems
    """    
    
   

    def __init__(self, fid):      
        
        self.hdf = h5py.File(fid)
        self.config = self.hdf[CONFIG]
        
        
        """
        
        #a list of subsystems
        self.subsystems = None
        self.config = config
        self.pbc = array(config["pbc"])
        self.timestep = 0
        
        # default values
        self.ndx = "main.ndx"
        self.mainselection = "Protein"
        
        if self.config.has_key("hdf") and nframe is None:
            self.hdf = h5py.File(self.config["hdf"], "w")
        else:
            self.hdf = None

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
        
        # number of data points in trajectory, md steps / output interval
        md_nsteps = int(config["md"]["nsteps"])/int(md.MD_config_get(config["md"], "nstout"))
        
        nrs = len(self.subsystems)
        
        # cg: nensembe x n segment x 3
        self.pos        = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))
        self.forces     = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))
        self.velocities = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))

        logging.info("pos {}".format(self.pos.shape))
        logging.info("frc {}".format(self.forces.shape))
        logging.info("vel {}".format(self.velocities.shape))

        if nframe is not None:
            self.read_frame(config["hdf"], nframe)
        """
        
    @property
    def struct(self):
        return self.hdf[CURRENT_TIMESTEP + "/" + STRUCT_PDB]
    
    @property
    def top(self):
        return self.hdf[CURRENT_TIMESTEP + "/" + TOPOL_TOP]
    
    def _begin_timestep(self):
        """
        The _begin_timestep and _end_timestep logic are modeled after OpenGL's glBegin and glEnd. 
        the "current_timestep" link should only exist between calls 
        to _begin_timestep and _end_timestep. The run may crash inbetween these
        calls, in this case, we assume whatever is the contents of this timestep
        is garbage and delete it.
        """
        # if there is a current timestep, we assume its garbage and delete it
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            del self.hdf[CURRENT_TIMESTEP]
        
        self.hdf.create_group(CURRENT_TIMESTEP)
            
        # prev_timestep could only have been created with a valid _end_timestep            
        src_files = self.hdf[PREV_TIMESTEP] if \
            self.hdf.id.links.exists(PREV_TIMESTEP) else \
            self.hdf[SRC_FILES]
        
        # link file data into current     
        for f in FILE_DATA_LIST:
            util.hdf_linksrc(self.hdf, CURRENT_TIMESTEP + "/" + f, src_files.name + "/" + f)
            
    def _end_timestep(self):
        """
        move current_timestep to timesteps/n
        """
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        prev = -1 if len(timesteps) == 0 else max(timesteps)
        finished = TIMESTEPS + "/" + str(prev+1)
        self.hdf.id.move(CURRENT_TIMESTEP, finished)
        
        # relink prev_timestep to the just finished ts.
        if self.hdf.id.links.exists(PREV_TIMESTEP):
            del self.hdf[PREV_TIMESTEP]
        self.hdf.id.links.create_soft(PREV_TIMESTEP, finished)
        
        self.hdf.flush()
            
            
    def _topology(self, config):
        """ 
        set up the topology.
        if top is specified in config, it is used, otherwise a new one is created.
        
        @postcondition: self.top, self.struct, self.posres 
        
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
            
        self.key_fromfile("struct", top["struct"])
        self.key_fromfile("top", top["top"])
        self.key_fromfile("posres", top["posres"])

        
        
    def hdf_write(self, name, value):
        """
        write a value to the output file using the given key at the current timestep.
        """
        
        if self.hdf is not None:
            keys = self.hdf.keys()
            if(keys.count(str(self.timestep))):
                grp = self.hdf[str(self.timestep)]
            else:
                grp = self.hdf.create_group(str(self.timestep))
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
    
            self.hdf.flush()
        
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
        self._begin_timestep()
        
        # md - runs md with current state, reads in md output and populates 
        # segments statistics. 
        self.minimize()
        
        self.thermalize()
        
        self.md()
        
        self.evolve()
        
        self._end_timestep()
        logging.info("completed step {}".format(self.timestep))
        
        
    def evolve(self):
        beta_t = float(self.config["beta_t"])
        D = diffusion(self.velocities)
        f = mean(self.forces, axis=0).flatten()
        Df = dot(D,f)
        
        self.hdf_write("DIFFUSION", D)
        
        for s in enumerate(self.subsystems):
            s[1].translate(Df[s[0]:s[0]+self.ncgs])
            
        self.hdf_write("ATOMIC_POS", self.universe.atoms.positions)
        
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
                "multi":self.pos.shape[0],
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
        mdr["multi"] = self.pos.shape[0]
        
        # run the md
        mdr = md.run_MD(System.md_dir, **mdr)
        
        self._processes_trajectories(mdr.trajectories)

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
        # set up a restrained md run
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
        
        self.hdf_write("ATOMIC_EQUILIBRIATED_POSITIONS", self.universe.atoms.positions)
        
        [s.equilibriated() for s in self.subsystems]
        
        logging.info("finished System.equilibriate()")
        
    def _processes_trajectories(self, trajectories):
        """
        reads each given atomic trajectory, extracts the
        coarse grained information, updates state variables with 
        this info, and saves it to the output file.
        
        @precondition: univserse is intialized with a topology
        compatible with the topologies
        
        """
        # zero the state variables (for this frame)
        self.pos[:] = 0.0
        self.forces[:] = 0.0
        self.velocities[:] = 0.0

        for fi, f in enumerate(trajectories):
            print(f)
            self.universe.load_new(f)
            for tsi, _ in enumerate(self.universe.trajectory):
                if tsi < self.velocities.shape[2]:
                    for si, s in enumerate(self.subsystems):
                        pos,vel,frc = s.frame()
                        self.pos       [fi,si,tsi,:] = pos
                        self.velocities[fi,si,tsi,:] = vel
                        self.forces    [fi,si,tsi,:] = frc
                        
                        if(tsi % 25 == 0):
                            print("processing frame {},\t{}%".format(tsi, 100.0*float(tsi)/float(self.velocities.shape[2])))
    
                        
                
            self.universe.trajectory.close()
            
        # done with trajectories, reload the starting (equilibriated) structure
        # need to do this as the md result will delete the trajectory files,
        # and if these are deleted while the universe has them open, bad things
        # happen.
        #self.universe.load_new(System.system_struct)
        
        # TODO
        # delete md output files...
            
        # done with files, divide by n frames to get average
        # self.pos[:,:,:] /= (self.velocities.shape[2])
        # self.forces[:,:,:] /= (self.velocities.shape[2])  
        
        self.hdf_write("POSITIONS", self.pos)  
        self.hdf_write("FORCES", self.forces)  
        self.hdf_write("VELOCITIES", self.velocities)
        
    def topology_changed(self):
        """
        Notify the system that the topology of the universe was changed. 
        This causes the system to generate a new topology file, 
        and notify all subsystems that the universe was changed.
        """
        pass
    
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
        conf = {"struct":self.universe, "top":self.top, 
                "ndx":self.ndx, "mainselection":self.mainselection}
        conf.update(self.config.get("minimize", {}))
        mn = md.minimize(struct=self.universe, top=self.top)
                
        # output of minimize looks like:
        # {'top': '/home/andy/tmp/1OMB/top/system.top', 'mainselection': '"Protein"', 'struct': '/home/andy/tmp/1OMB/em/em.pdb'}
        
        self.struct = mn["struct"]
        self.mainselection = mn["mainselection"]
        self.top = mn["top"]
        print(mn)
        
        self.universe.load_new(self.struct)
        
        self.hdf_write("ATOMIC_MINIMIZED_POSITIONS", self.universe.atoms.positions)
        [s.minimized() for s in self.subsystems]
            
        
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
        
    def key_tofile(self, key, f, timestep=None):
        """
        """
        timestep = self.timestep if timestep is None else timestep
        data = self.hdf["{}/{}".format(timestep, key)][()]
        data.tofile(f)
        
    def key_fromfile(self, key, f, timestep=None):
        """
        """
        timestep = self.timestep if timestep is None else timestep
        self.hdf["{}/{}".format(timestep,key)] = fromfile(f, dtype=uint8)
        
        
def __get_class( kls ):
    """
    given a fully qualified class name, i.e. "datetime.datetime", 
    this loads the module and returns the class type. 
    
    the ctor on the class type can then be called to create an instance of the class.
    """
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__( module )
    for comp in parts[1:]:
        m = getattr(m, comp)            
    return m
        
        
        
    
        
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
    "md":{"nsteps":50000},
    "multi":100,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "top":"C60.top",
    "hdf":"out.hdf"
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
    "multi":20,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "top":"topol.top"
    } 

Au = { 
    'temperature' : 300.0, 
    'struct': '/home/andy/tmp/Au/100.0_10.0.sol.pdb',
    'top': '/home/andy/tmp/Au/100.0_10.0.top',
    "subsystems" : [subsystems.RigidSubsystemFactory, "not resname SOL"],
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize":{"nsteps":1000},
    "md":{"nsteps":250000},
    "multi":1,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "hdf":"100.0_10.0.hdf"
    } 



def test(fbase):
    print(os.getcwd())
    
    logger = logging.getLogger()
    logger.handlers[0].stream.close()
    logger.removeHandler(logger.handlers[0])
    
    handler = logging.StreamHandler()



    handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(filename)s, %(lineno)d, %(funcName)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    #logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s', filename=fbase + ".log",level=logging.DEBUG)
    
    Au2["struct"] = Au2["struct"].format(fbase)
    Au2["top"] = Au2["top"].format(fbase)
    Au2["hdf"] = Au2["hdf"].format(fbase)
    
    s=System(Au2)
    
    for ss in s.subsystems:
        print(ss.atoms.masses())
        
    #s._write_system_struct()
    
    s._processes_trajectories(["/home/andy/Au/{}.trr".format(fbase)])

Au2 = { 
    'box' : [50.0, 50.0, 50.0],      
    'temperature' : 300.0, 
    'struct': '/home/andy/tmp/1OMB/1OMB.pdb',
    "subsystem_class" : "dms2.subsystems.RigidSubsystemFactory",
    "subsystem_select": "not resname SOL",
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize_steps":1000,
    "md_steps":250000,
    "multi":1,
    "equilibriate_steps":1000,
    "solvate":True,
    "hdf":"{}.hdf"
    } 
    
def ctest():
    create_config(fid="test.hdf", **Au2)    

def create_config(fid,
                  struct,
                  box,
                  top = None,
                  temperature = 300,
                  subsystem_class = "dms2.subsystems.RigidSubsystemFactory",
                  subsystem_select = "not resname SOL",
                  cg_steps = 50,
                  beta_t  = 10.0,
                  minimize_steps = 1000,
                  md_steps = 1000,
                  multi = 1,
                  equilibriate_steps = 1000,
                  solvate = False,
                  posres=None,
                  ndx=None,
                  **kwargs):
    
    import gromacs
    
    with h5py.File(fid, "w") as hdf:

        conf = hdf.create_group("config").attrs
        src_files = hdf.create_group("src_files")

        def filedata_fromfile(keyname, filename):
            try:
                del src_files[str(keyname)]
            except KeyError:
                pass
            src_files[str(keyname)] = fromfile(struct, dtype=uint8)

        def int_attr(keyname, value):
            try:
                conf[keyname] = int(value)
            except Exception, e:
                print("error, {} {} must be a numeric value".format(keyname, value))
                raise e
        try:
            box = array(box) 
            print("periodic boundary conditions: {}".format(box))
            box /= 10.0 # convert Angstrom to Nm
            conf["box"] = box
        except Exception, e:
            print("error reading periodic boundary conditions")
            raise e

        try:
            _ = __get_class(subsystem_class)
            conf["subsystem_class"] = subsystem_class
            print("subsystem_class: {}".format(subsystem_class))

        except Exception, e:
            print("error creating subsystem_class, {}".format(e))
            raise e

        try:
            conf["temperature"] = float(temperature)
        except Exception, e:
            print("error, temperature {} must be a numeric value".format(temperature))
            raise e

        int_attr("cg_steps", cg_steps)
        int_attr("beta_t", beta_t)
        int_attr("minimize_steps", minimize_steps)
        int_attr("md_steps", md_steps)
        int_attr("multi", multi)
        int_attr("equilibriate_steps", equilibriate_steps)
        int_attr("solvate", solvate)

        # check struct
        try:
            gromacs.gmxcheck(f=struct)
            print("structure file {} appears OK".format(struct))
            filedata_fromfile("struct.pdb", struct)
        except Exception, e:
            print("structure file {} is not valid".format(struct))
            raise e

        # make a top if we don't have one
        if top is None:
            print("attempting to auto-generate a topology...")
            with md.topology(struct=struct, protein="protein", posres=posres) as top:
                # topology returns:
                # {'top': '/home/andy/tmp/Au/top/system.top', 
                # 'dirname': 'top', 
                # 'posres': 'protein_posres.itp', 
                # 'struct': '/home/andy/tmp/Au/top/protein.pdb'}

                print("succesfully auto-generated topology")

                filedata_fromfile("topol.top", top["top"])
                filedata_fromfile("posres.itp", top["posres"])

                if solvate:
                    with md.solvate(box=box, **top) as sol:
                        # solvate returns 
                        # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
                        # 'mainselection': '"Protein"', 
                        # 'struct': '/home/andy/tmp/Au/solvate/solvated.pdb', 
                        # 'qtot': 0})
                        filedata_fromfile("index.ndx", sol["ndx"])
                        filedata_fromfile("struct.pdb", sol["struct"])
                        filedata_fromfile("topol.top", sol["top"])

        if posres is not None:
            filedata_fromfile("posres.itp", posres)

        if ndx is not None:
            filedata_fromfile("index.ndx", ndx)

        # link required file data
        files = hdf.create_group("files")
        for f in ["struct.pdb", "topol.top", "index.ndx", "posres.itp"]:
            files[f] = h5py.SoftLink("/src_files/" + f)
            
        hdf.create_group("timesteps")
        
                    
            
            
        
        
    
    """

    'top': '/home/andy/tmp/Au/{}.top',
    "subsystem.class" : "dms2.subsystems.RigidSubsystemFactory",
    "subsystem.args": "not resname SOL",
    "cg_steps":150,
    "beta_t":10.0,
    "top_args": {},
    "minimize":{"nsteps":1000},
    "md":{"nsteps":250000},
    "multi":1,
    "equilibriate":{"nsteps":1000},
    "solvate":False,
    "hdf":"{}.hdf"
    """
    
    
        
    
    

def main():
    ctest()
    """
    print(os.getcwd())
    
    s=System(C60)
    
    
    
    s.minimize()
    
    s.equilibriate()
    
    #s.test()
    s.md()
    """
    


    
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
    ctest()


        


    

