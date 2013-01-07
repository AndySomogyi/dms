"""
@group environment variables:
    DMS_DEBUG: if set to , temporary directores are NOT deleted, this might be usefull
    for debugging MD issues.

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
import dynamics

import h5py #@UnresolvedImport
import config
import md
import diffusion
import subsystems
import util
import time
import datetime


# change MDAnalysis table to read carbon correctly
import MDAnalysis.topology.tables
MDAnalysis.topology.tables.atomelements["C0"]="C"


CURRENT_TIMESTEP = "current_timestep" 
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
    ATOMIC_STARTING_POSITIONS = "atomic_starting_positions"
    ATOMIC_FINAL_POSITIONS = "atomic_final_positions"
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
        self.__create_property(Timestep.ATOMIC_STARTING_POSITIONS)
        self.__create_property(Timestep.ATOMIC_FINAL_POSITIONS)
        self.__create_property(Timestep.TIMESTEP_END)
        self.__create_property(Timestep.TIMESTEP_BEGIN)
        self.__create_property(Timestep.CG_POSITIONS)
        self.__create_property(Timestep.CG_VELOCITIES)
        self.__create_property(Timestep.CG_FORCES)
        
    @property
    def timestep(self):
        """
        the index of the timestep
        """
        return int(self._group.name.split("/")[-1])
        
        
    def create_universe(self):
        data = self._group[STRUCT_PDB][()]
        
        
        with tempfile.NamedTemporaryFile(suffix=".pdb") as f:  
            # EXTREMLY IMPORTANT to flush the file, 
            # NamedTemporaryFile returns an OPEN file, and the array writes to this file, 
            # so we need to flush it, as this file handle remains open. Then the MDAnalysis
            # universe opens another handle and reads its contents from it. 
            data.tofile(f.file)
            f.file.flush()
            u = MDAnalysis.Universe(f.name)
            return u
        
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
    
    # define the __slots__ class variable, this helps preven typos by raising an error if a
    # ivar is set that is not one of these here. 
    __slots__ = ["hdf", "config", "universe", "_box", "ncgs", "subsystems", "cg_positions", "cg_velocities", "cg_forces"]
    
    def __init__(self, fid):      
        
        self.hdf = h5py.File(fid)
        self.config = self.hdf[CONFIG].attrs
        self._box = self.config[BOX]
        
        # if there is a current timestep, we assume its garbage and delete it
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            logging.info("found previous \"current_timestep\" key, this means that a previous simulation likely crashed, deleting it.")
            del self.hdf[CURRENT_TIMESTEP]
            
        # load the universe object from either the last timestep, or from the src_files
        # its expensive to create a universe, so keep it around for the lifetime
        # of the system
        self.universe = self._create_universe()
        
        # load the subsystems
        # this list will remain constant as long as the topology remains constant.
        logging.info("creating subsystems")
        factory = util.get_class(self.config[SUBSYSTEM_FACTORY])
        self.ncgs, self.subsystems = factory(self, self.config[SUBSYSTEM_SELECTS], *self.config[SUBSYSTEM_ARGS])
        logging.debug("using {} subsystems and {} coarse grained vars".format(self.subsystems, self.ncgs))
        
        # notify subsystems, we have a new universe
        [s.universe_changed(self.universe) for s in self.subsystems]
        
        md_nensemble = self.config[MULTI]
        
        # md args dictionary
        mdd = dict(zip(self.config[MD_ARGS + KEYS], self.config[MD_ARGS + VALUES]))
        # number of data points in trajectory, md steps / output interval
        md_nsteps = int(self.config[MD_STEPS])/int(mdd[NSTXOUT])
        
        # number of subsystems
        nrs = len(self.subsystems)
        
        # cg: nensembe x n segment x n_step x n_cg
        self.cg_positions  = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))
        self.cg_forces     = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))
        self.cg_velocities = zeros((md_nensemble,nrs,md_nsteps,self.ncgs))

        logging.info("pos {}".format(self.cg_positions.shape))
        logging.info("frc {}".format(self.cg_forces.shape))
        logging.info("vel {}".format(self.cg_velocities.shape))
        
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
    
    @property
    def beta_t(self):
        return float(self.config[BETA_T])
    
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
    
    def __last_timestep(self):
        """
        Gets the last completed timestep, or None if this is the first timestep.
        
        Public API should only call the "timesteps" generator.
        """
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        if len(timesteps):
            return Timestep(self.hdf[TIMESTEPS + "/" + str(max(timesteps))])
        else:
            return None
        
        
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
            logging.info("found previous \"current_timestep\" key, this means that end_timestep was not called.")
            del self.hdf[CURRENT_TIMESTEP]
            
        # create a new current_timestep group, starting time now.
        current_group = self.hdf.create_group(CURRENT_TIMESTEP)
        self.current_timestep.timestep_begin = time.time()
            
        last = self.__last_timestep()             
        if last:
            # link the starting positions to the previous timesteps final positions
            current_group.id.links.create_soft(Timestep.ATOMIC_STARTING_POSITIONS, 
                                      last._group.name + "/" + Timestep.ATOMIC_FINAL_POSITIONS)
            src_files = last._group
        else:
            # this is the first timestep, so set start positions to the starting
            # universe struct.
            current_group[Timestep.ATOMIC_STARTING_POSITIONS] = self.universe.atoms.positions
            src_files = self.hdf[SRC_FILES]
        
        # link file data into current     
        for f in FILE_DATA_LIST:
            util.hdf_linksrc(self.hdf, CURRENT_TIMESTEP + "/" + f, src_files.name + "/" + f)
            
            
    def end_timestep(self):
        """
        move current_timestep to timesteps/n
        """
        # done with timestep, throws an exception is we do not
        # have a current timestep.
        self.current_timestep.timestep_end = time.time()
        
        # find the last timestep, and set this one to the next one, and move it there.
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        prev = -1 if len(timesteps) == 0 else max(timesteps)
        finished = TIMESTEPS + "/" + str(prev+1)
        self.hdf.id.move(CURRENT_TIMESTEP, finished)
        
        self.hdf.flush()
        
        logging.info("completed timestep {}".format(prev+1))
        
    @property
    def current_timestep(self):
        return Timestep(self.hdf[CURRENT_TIMESTEP])
    
    @property
    def timesteps(self):
        # get all the completed timesteps, the keys are usually out of order, so
        # have to sort them
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        timesteps.sort()
        
        for ts in timesteps:
            yield Timestep(self.hdf[TIMESTEPS + "/" + str(ts)])
        

    def _create_universe(self, key = None):
        """
        Creates a universe object from the most recent timestep, or if this is the first 
        timestep, loads from the 'src_files' key.
        """

        last = self.__last_timestep()
        if last:
            logging.info("loading universe from most recent timestep of {}".format(last._group.name))
            return last.create_universe()
        else:
            logging.info("_universe_from_hdf, struct.pdb not in current frame, loading from src_files")
            data = array(self.hdf[SRC_FILES + "/" + STRUCT_PDB])  
    
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
        last = self.__last_timestep()
        start = last.timestep + 1 if last else 0
        del last
        end = int(self.config[CG_STEPS])
        
        logging.info("running timesteps {} to {}".format(start, end))
        
        for _ in range(start, end):
            self.step()
            
        logging.info("completed all {} timesteps".format(end-start))
   

    def step(self):
        """
        performs a single time step
        """
        self.begin_timestep()
        
        # md - runs md with current state, reads in md output and populates 
        # segments statistics. 
        self.minimize()
        
        self.equilibriate()
        
        self.md()
        
        self.evolve()
        
        self.end_timestep()
        
        
        
    def evolve(self):
        
        # forward euler
        # result = coordinate_ops + beta() * dt * Df
        Df = dynamics.diffusion_force(self) * self.beta_t
        
        # Df is a column vector, change it to a row vector, as the subsystems
        # expect a length(ncg) row vector. 
        Df = Df.transpose()
        
        for i, s in enumerate(self.subsystems):
            s.translate(Df[i*self.ncgs:self.ncgs])
            
        self.current_timestep.atomic_final_positions = self.universe.atoms.positions
        
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
                               multi=self.config[MULTI], \
                               **dict(zip(self.config[MD_ARGS + KEYS], self.config[MD_ARGS + VALUES])))
        
    def equilibriate(self):
        with self.setup_equilibriate() as eqsetup:
            mdres = md.run_md(eqsetup.dirname, **eqsetup)
            self.universe.load_new(mdres.structs[0])
        
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
        self.cg_positions[:] = 0.0
        self.cg_forces[:] = 0.0
        self.cg_velocities[:] = 0.0
        
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
                    if tsi < self.cg_velocities.shape[2]:
                        for si, s in enumerate(self.subsystems):
                            pos,vel,frc = s.frame()
                            self.cg_positions       [fi,si,tsi,:] = pos
                            self.cg_velocities[fi,si,tsi,:] = vel
                            self.cg_forces    [fi,si,tsi,:] = frc
                            
                            if(tsi % 25 == 0):
                                print("processing frame {},\t{}%".format(tsi, 100.0*float(tsi)/float(self.cg_velocities.shape[2])))
            
            # done with trajectories, load original contents of universe back
            self.universe.load_new(tmp.name)
            
        # write the coarse grained pos, vel and forces to the current timestep.
        timestep = self.current_timestep
        timestep.cg_positions = self.cg_positions
        timestep.cg_velocities = self.cg_velocities
        timestep.cg_forces = self.cg_forces
            

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
        with md.minimize(struct=self.universe, \
                         top=self.top, \
                         posres = self.posres, \
                         nsteps=self.config[MN_STEPS], \
                          **dict(zip(self.config[MN_ARGS + KEYS], self.config[MN_ARGS + VALUES]))) as mn:

            print(mn)
        
            self.universe.load_new(mn["struct"])
            
        self.current_timestep.atomic_minimized_positions = self.universe.atoms.positions
        [s.minimized() for s in self.subsystems]
        
        
    def tofile(self,traj):
        """
        Write the system to a conventional MD file format, either pdb, or trr. 
        
        if traj ends with 'pdb', the starting structure is saved as a pdb, otherwise, 
        all the frames are written to a trajectory (trr). This is usefull for VMD
        where one could perform:
            sys.tofile("somefile.pdb")
            sys.tofile("somefile.trr"),
        then call vmd with these file args.
        """
        universe = None
        writer = None
        ext = os.path.splitext(traj)[-1]
        if ext.endswith("pdb"):
            for ts in self.timesteps:
                universe = ts.create_universe()
                writer = MDAnalysis.Writer(traj,numatoms=len(universe.atoms))
                writer.write(universe)
                return
        else:
            for ts in self.timesteps:
                if universe is None:
                    universe = ts.create_universe()
                    writer = MDAnalysis.Writer(traj,numatoms=len(universe.atoms))
                else:
                    universe.atoms.positions = ts.atomic_starting_positions
                writer.write(universe)
                
    def visualize(self):
        dirname = tempfile.mkdtemp()
        struct = dirname + "/struct.pdb"
        traj = dirname + "/traj.trr"
        
        
        self.tofile(struct)
        self.tofile(traj)
        
        os.system("vmd {} {}".format(struct, traj))
        
        
        
        
            
            
        
def testsys():
    return System('test.hdf')
    
        
from numpy.fft import fft, ifft, fftshift

"""
def diffusion(vel, D):
    
    calculate the matrix of diffusion coeficients.
    @param vel: a #ensembles x #subsytems x #frames x #cg array
    @return: a (#subsytems x #cg) x (#subsytems x #cg) array
    
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
"""


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


Au2 = { 
    'box' : [50.0, 50.0, 50.0],      
    'temperature' : 300.0, 
    'struct': '/home/andy/tmp/1OMB/1OMB.pdb',
    "subsystem_select": "not resname SOL",
    "cg_steps":5,
    "beta_t":10.0,
    "top_args": {},
    "mn_steps":5,
    "eq_steps":50,
    "md_steps":50,
    "multi":4,
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
                  cg_steps = 10,
                  beta_t  = 10.0,
                  mn_steps = 500,
                  md_steps = 100,
                  multi = 1,
                  eq_steps = 10,
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
            gromacs.gmxcheck(f=struct) #@UndefinedVariable
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


        


    

