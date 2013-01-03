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
    
    The "/timesteps" group has a series of timestep groups named 0, 1, ... N.
    Each one of these timestep subgroups contains the following data 
    
        struct.pdb: this is typically a soft link to the struct.pdb 
        in "/src_files". This is only used as a way to store the non-coordinate
        attributes such as segment and residue info. The coordinates in this
        pdb are NOT USED to store coordinate info.
        
        
    
    
        
    
        
            
    
    
    
                 
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
import tempfile
import MDAnalysis

import h5py #@UnresolvedImport
import config
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
CG_STEPS = "cg_steps"
BETA_T = "beta_t"
MN_STEPS = "mn_steps"
EQ_STEPS = "eq_steps"
MD_STEPS = "md_steps"

# All values are stored in the hdf file, however there is no direct
# way to store a python dict in hdf, so we need to store it as
# a set of key / value lists.
# This is a bit hackish, but to store them as lists, the following
# 'key names' will have '_keys' and '_values' automatically appended
# to them, then the corresponding keys / values will be stored as 
# such. 
MN_ARGS = "mn_args"
EQ_ARGS = "eq_args"
MD_ARGS = "md_args"
KEYS="_keys"
VALUES="_values"
MULTI= "multi"
SOLVATE="solvate"
POSRES_ITP="posres.itp"
SUBSYSTEM_FACTORY = "subsystem_factory"
SUBSYSTEM_SELECTS = "subsystem_selects"
SUBSYSTEM_ARGS = "subsystem_args"
NSTXOUT = "nstxout"
NSTVOUT = "nstvout"
NSTFOUT = "nstfout"
BOX = "box"

class Timestep(object):
    """
    each timestep is storred in a hdf group, this class wraps
    the group and provides properties to access the values.
    """
    ATOMIC_MINIMIZED_POSITIONS = "atomic_minimized_positions"
    ATOMIC_EQUILIBRIATED_POSITIONS = "atomic_equilibriated_positions"
    TIMESTEP_BEGIN = "timestep_begin"
    TIMESTEP_END = "timestep_end" 
    CG_POSITIONS = "cg_positions"
    CG_VELOCITIES = "cg_velocities"    
    CG_FORCES = "cg_forces"
    
    def __create_property(self, name):
        def getter(self):
            return self._group[name][()]
        def setter(self,val):
            try:
                del self._group[name]
            except KeyError:
                pass
            self._group[name] = val
            self._group.file.flush()
            
        # construct property attribute and add it to the class
        setattr(self.__class__, name, property(fget=getter, fset=setter))
    
    def __init__(self, group):
        self._group = group
        self.__create_property(Timestep.ATOMIC_MINIMIZED_POSITIONS)
        self.__create_property(Timestep.ATOMIC_EQUILIBRIATED_POSITIONS)
        self.__create_property(Timestep.TIMESTEP_END)
        self.__create_property(Timestep.TIMESTEP_BEGIN)
        self.__create_property(Timestep.CG_POSITIONS)
        self.__create_property(Timestep.CG_VELOCITIES)
        self.__create_property(Timestep.CG_FORCES)
        
class System(object):


    """
    @ivar subsystems: a list of subsystems, remains constant so long 
                      as the topology does not change.
                      
    @ivar qtot = total charge, set by solvate
    @ivar mainselection = set by solvate, defaults to "Protein",
    @ivar ndx = the index file, set by solvate, defaults to 'main.ndx'
    
    @ivar subsystems: a list of subsystems
    
    @ivar _universe: an MDAnalysis Universe object that maintains the atomic 
    state. 
    """    
    
    def __init__(self, fid):      
        
        self.hdf = h5py.File(fid)
        self.config = self.hdf[CONFIG].attrs
        self.universe = self._universe_from_hdf()
        self._box = self.config[BOX]
        
        # load the subsystems
        # this list will remain constant as long as the topology remains constant.
        logging.info("creating subsystems")
        factory = util.get_class(self.config[SUBSYSTEM_FACTORY])
        self.ncgs, self.subsystems = factory(self, self.config[SUBSYSTEM_SELECTS], *self.config[SUBSYSTEM_ARGS])
        logging.debug("using {} subsystems and {} coarse grained vars".format(self.subsystems, self.ncgs))
        
        # notify subsystems, we have a new universe
        [s.universe_changed(self.universe) for s in self.subsystems]
        
        md_nensemble = self.config[MULTI]
        
        mdd = dict(zip(self.config[MD_ARGS + KEYS], self.config[MD_ARGS + VALUES]))
        # number of data points in trajectory, md steps / output interval
        md_nsteps = int(self.config[MD_STEPS])/int(mdd[NSTXOUT])
        
        nrs = len(self.subsystems)
        
        # cg: nensembe x n segment x 3
        self.pos        = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))
        self.forces     = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))
        self.velocities = zeros((md_nensemble,nrs,md_nsteps, self.ncgs))

        logging.info("pos {}".format(self.pos.shape))
        logging.info("frc {}".format(self.forces.shape))
        logging.info("vel {}".format(self.velocities.shape))
        
    @property
    def struct(self):
        return self._get_file_data(STRUCT_PDB)
    
    @property
    def top(self):
        return self._get_file_data(TOPOL_TOP)
    
    @property
    def posres(self):
        return self._get_file_data(POSRES_ITP)
    
    @property
    def box(self):
        return self._box
    
    def _get_file_data(self, file_key):
        """
        finds a file stored as an hdf key. This first looks if there is a 'current_timestep', 
        if so uses it, otherwise, looks in 'src_files'. 
        """
        #TODO imporove error checking and handling.
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            file_key = CURRENT_TIMESTEP + "/" + file_key
        else:
            file_key = SRC_FILES + "/" + file_key
        return self.hdf[file_key]
        
    
    def begin_timestep(self):
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
            
    def end_timestep(self):
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
        
    @property
    def current_timestep(self):
        return Timestep(self.hdf[CURRENT_TIMESTEP])
        
    def _universe_from_hdf(self, key = None):
        """
        returns a Universe object from a structure file stored as a 
        blob in the hdf file.
        """

        data = None
        
        if key is None:
            # check if restart, use from current timestep
            if self.hdf.id.links.exists(CURRENT_TIMESTEP):
                logging.debug("_universe_from_hdf, found struct.pdb in current frame, loading.")
                data = array(self.hdf[CURRENT_TIMESTEP + "/" + STRUCT_PDB])
            else:
                logging.debug("_universe_from_hdf, struct.pdb not in current frame, loading from src_files")
                data = array(self.hdf[SRC_FILES + "/" + STRUCT_PDB])  
        else:
            data = array(self.hdf[key])
        
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:  
            # EXTREMLY IMPORTANT to flush the file, 
            # NamedTemporaryFile returns an OPEN file, and the array writes to this file, 
            # so we need to flush it, as this file handle remains open. Then the MDAnalysis
            # universe opens another handle and reads its contents from it. 
            data.tofile(f.file)
            f.file.flush()
            u = MDAnalysis.Universe(f.name)
            return u
            

        
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
        


        

    def setup_equilibriate(self):
        """
        setup an equilibriation md run using the contents of the universe, but do not actually run it.

        @return an MDManager object loaded with the trr to run an equilibriation.
        """

        logging.info("setting up equilibriation...")
        
        return md.setup_md(struct=self.universe, \
                               top=self.top, \
                               posres = self.posres, \
                               nsteps=self.config[EQ_STEPS], \
                               dirname="eq_test", \
                               **dict(zip(self.config[EQ_ARGS + KEYS], self.config[EQ_ARGS + VALUES])))
    
    def setup_md(self):
        """
        setup an equilibriation md run using the contents of the universe, but do not actually run it.

        @return an MDManager object loaded with the trr to run an equilibriation.
        """

        logging.info("setting up equilibriation...")
        
        return md.setup_md(struct=self.universe, \
                               top=self.top, \
                               posres = self.posres, \
                               nsteps=self.config[MD_STEPS], \
                               dirname="md_test", \
                               multi=self.config[MULTI], \
                               **dict(zip(self.config[MD_ARGS + KEYS], self.config[MD_ARGS + VALUES])))
        
    def equilibriate(self):
        with self.setup_equilibriate() as eqsetup:
            mdres = md.run_md(eqsetup.dirname, **eqsetup)
            self.universe.load_new(mdres.struct)
        
        self.current_timestep.atomic_equilibriated_positions = self.universe.atoms.positions
        [s.equilibriated() for s in self.subsystems]
        
    def md(self):
        with self.setup_md() as mdsetup:
            mdres = md.run_md(mdsetup.dirname, **mdsetup)            
            self._processes_trajectories(mdres.trajectories)
        

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
        
        # save the universe to a tmp file to load back once the trajectories 
        # are processed.
        with tempfile.NamedTemporaryFile(suffix=".pdb") as tmp:  
            writer = MDAnalysis.Writer(tmp.name)
            writer.write(self.universe)
            writer.close()
            del writer
            
            # universe is saved, process trajectories
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
            
            # done with trajectories, load original contents of universe back
            self.universe.load_new(tmp.name)
            
        timestep = self.current_timestep
        timestep.cg_positions = self.pos
        timestep.cg_velocities = self.velocities
        timestep.cg_forces = self.forces
            

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
        
        @postcondition: 
        self.universe is loaded with the minimized structure
        all subsystems are notified.
        """
        with md.minimize(struct=self.universe, top=self.top, nsteps=self.config[MN_STEPS]) as mn:

            print(mn)
        
            self.universe.load_new(mn["struct"])
            
        self.current_timestep.atomic_minimized_positions = self.universe.atoms.positions
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

"""

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
"""

Au2 = { 
    'box' : [50.0, 50.0, 50.0],      
    'temperature' : 300.0, 
    'struct': '/home/andy/tmp/1OMB/1OMB.pdb',
    "subsystem_select": "not resname SOL",
    "cg_steps":10,
    "beta_t":10.0,
    "top_args": {},
    "minimize_steps":100,
    "md_steps":50,
    "multi":4,
    "equilibriate_steps":1000,
    "solvate":False,
    } 
    
def ctest(pdb, hdf):
    Au2['struct'] = pdb
    create_config(fid=hdf, **Au2)

DEFAULT_MD_ARGS = { "mdp":config.templates["md_CHARMM27.mdp"],  # the default mdp template \
                    "nstxout": 10,    # trr pos
                    "nstvout": 10,    # trr veloc
                    "nstfout": 10,    # trr forces
                    }

DEFAULT_EQ_ARGS = { "mdp":config.templates["md_CHARMM27.mdp"],  # the default mdp template 
                    "define":"-DPOSRES" # do position restrained md for equilibriation
                    }

def create_config(fid,
                  struct,
                  box,
                  top = None,
                  temperature = 300,
                  subsystem_factory = "dms2.subsystems.RigidSubsystemFactory",
                  subsystem_selects = ["not resname SOL"],
                  subsystem_args = [],
                  cg_steps = 50,
                  beta_t  = 10.0,
                  mn_steps = 500,
                  md_steps = 100,
                  multi = 1,
                  eq_steps = 100,
                  mn_args = {},
                  eq_args = DEFAULT_EQ_ARGS,
                  md_args = DEFAULT_MD_ARGS,
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
            src_files[str(keyname)] = fromfile(filename, dtype=uint8)


        # create an attr key /  value in the config attrs
        def attr(keyname, typ, value):
            if value is not None:
                try:
                    if typ is dict:
                        conf[keyname + KEYS] = value.keys()
                        conf[keyname + VALUES] = value.values()
                    else:
                        conf[keyname] = typ(value)
                    print("config[{}] = {}".format(keyname, value))
                except Exception, e:
                    print("error, could not convert \"{}\" with value of \"{}\" to an {} type".
                          format(keyname, value, typ))
                    raise e
                
                
        try:
            box = array(box) 
            print("periodic boundary conditions: {}".format(box))
            conf[BOX] = box
        except Exception, e:
            print("error reading periodic boundary conditions")
            raise e

        try:
            factory = util.get_class(subsystem_factory)
            test_ncgs, test_ss = factory(None, subsystem_selects, *subsystem_args)
            if len(test_ss) == len(subsystem_selects):
                print("subsystem factory {} produces correct number of subsystems from {}".
                      format(subsystem_factory, subsystem_selects))
                print("will use {} coarse grained variables".format(test_ncgs))
            else:
                raise ValueError("subsystem factory {} is valid, but does NOT produce correct number of subsystems from {}".
                      format(subsystem_factory, subsystem_selects))
            conf[SUBSYSTEM_FACTORY] = subsystem_factory
            conf[SUBSYSTEM_SELECTS] = subsystem_selects
            conf[SUBSYSTEM_ARGS] = subsystem_args
            print("{}: {}".format(SUBSYSTEM_FACTORY, subsystem_factory))

        except Exception, e:
            print("error creating subsystem_class, {}".format(e))
            raise e

        try:
            conf["temperature"] = float(temperature)
        except Exception, e:
            print("error, temperature {} must be a numeric value".format(temperature))
            raise e

        attr(CG_STEPS, int, cg_steps)
        attr(BETA_T, int, beta_t)
        attr(MN_STEPS, int, mn_steps)
        attr(MD_STEPS, int, md_steps)
        attr(MULTI, int, multi)
        attr(EQ_STEPS, int, eq_steps)
        attr(SOLVATE, int, solvate)

        attr(MN_ARGS, dict, mn_args)
        attr(EQ_ARGS, dict, eq_args)
        attr(MD_ARGS, dict, md_args)

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
                
                print('pwd', os.getcwd())
                print('top', top)

                filedata_fromfile(TOPOL_TOP, top["top"])
                filedata_fromfile(POSRES_ITP, top["posres"])
                filedata_fromfile(STRUCT_PDB, top["struct"])

                if solvate:
                    # convert Angstrom to Nm, GROMACS works in Nm, and
                    # we use MDAnalysis which uses Angstroms
                    with md.solvate(box=box/10.0, **top) as sol:
                        # solvate returns 
                        # {'ndx': '/home/andy/tmp/Au/solvate/main.ndx', 
                        # 'mainselection': '"Protein"', 
                        # 'struct': '/home/andy/tmp/Au/solvate/solvated.pdb', 
                        # 'qtot': 0})
                        filedata_fromfile(INDEX_NDX, sol["ndx"])
                        filedata_fromfile(STRUCT_PDB, sol["struct"])
                        filedata_fromfile(TOPOL_TOP, sol["top"])

        if posres is not None:
            filedata_fromfile("posres.itp", posres)

        if ndx is not None:
            filedata_fromfile("index.ndx", ndx)

            
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
    
    import dms2.subsystems
    #ctest()
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
    
def test_md():
    sys = System("test.hdf")
    
    sys.equilibriate()
    

if __name__ == "__main__":
    if len(sys.argv) == 3:
        ctest(sys.argv[1], sys.argv[2])
    else:
    #import subsystems
        s=System("test.hdf")
        s._begin_timestep()
        s.minimize()


        


    

