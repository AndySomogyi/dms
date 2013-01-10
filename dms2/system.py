"""
@group Units: As dms2 uses MDAnalysis, it therefore uses the MDAnalysis units which are:
    force: kJ/(mol*A) - Note that MDA will convert forces to native units (kJ/(mol*A), even
    though gromcas uses kJ/(mol*nm).
    position: Angstroms,
    velocity: Angstrom/ps - velocities are automatically converted to MDAnalysis units 
    (i.e. from Gromacs nm/ps to Angstrom/ps in MDAnalysis)
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
logging.basicConfig(format='%(asctime)s:%(levelname)s:%(funcName)s:%(message)s', filename='dms.log',level=logging.DEBUG)
import sys
import os

from numpy import array, zeros, transpose, dot, reshape, \
                  average, arange, sqrt, linalg, conjugate, \
                  real, correlate, newaxis, sum, mean, min, max, \
                  where, pi, arctan2, sin, cos, fromfile, uint8
import numpy.random
import tempfile
import MDAnalysis #@UnusedImport
import dynamics

import h5py #@UnresolvedImport
import config
import md
import util
import time
import datetime
import collections


# change MDAnalysis table to read carbon correctly
import MDAnalysis.topology.tables
MDAnalysis.topology.tables.atomelements["C0"]="C"

"""
Botlzmann's constant in kJ/mol/K
"""
KB = 0.0083144621


CURRENT_TIMESTEP = "current_timestep" 
SRC_FILES = "src_files"
FILE_DATA_LIST = ["struct.pdb", "topol.top", "index.ndx", "posres.itp"]
TIMESTEPS = "timesteps"
CONFIG = "config"
STRUCT_PDB = "struct.pdb"
TOPOL_TOP = "topol.top"
INDEX_NDX = "index.ndx"
CG_STEPS = "cg_steps"
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
TEMPERATURE="temperature"
DT = "dt"

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
    CG_TRANSLATE = "cg_translate"
    
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
        self.__create_property(Timestep.CG_TRANSLATE)
        
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
    
    def flush(self):
        self._group.file.flush()
        
class System(object):


    """
    @ivar subsystems: a list of subsystems, remains constant so long 
                      as the topology does not change.
    """    
    
    # define the __slots__ class variable, this helps preven typos by raising an error if a
    # ivar is set that is not one of these here. 
    __slots__ = ["hdf", "config", "universe", "_box", "ncgs", "subsystems", "cg_positions", "cg_velocities", "cg_forces"]
    
    def __init__(self, fid):      
        
        self.hdf = h5py.File(fid)
        self.config = self.hdf[CONFIG].attrs
        self._box = self.config[BOX]
        
        # if there is a current timestep, keep it around for debugging purposes
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            print("WARNING, found previous \"current_timestep\" key, this means that a previous simulation likely crashed")
            logging.warn("found previous \"current_timestep\" key, this means that a previous simulation likely crashed")

            
        # load the universe object from either the last timestep, or from the src_files
        # its expensive to create a universe, so keep it around for the lifetime
        # of the system
        self.universe = self._create_universe()
        
        # load the subsystems
        # this list will remain constant as long as the topology remains constant.
        logging.info("creating subsystems")
        factory = util.get_class(self.config[SUBSYSTEM_FACTORY])
        self.ncgs, self.subsystems = factory(self, self.config[SUBSYSTEM_SELECTS], *self.config[SUBSYSTEM_ARGS])
        logging.debug("using {} cg variables for each {} subsystems".format(self.ncgs, len(self.subsystems)))
        
        # notify subsystems, we have a new universe
        [s.universe_changed(self.universe) for s in self.subsystems]
        
        md_nensemble = self.config[MULTI]
        
        # number of data points in trajectory, md steps / output interval
        md_nsteps = int(self.config[MD_STEPS])/int(self.md_args[NSTXOUT])
        
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
    def temperature(self):
        return float(self.config[TEMPERATURE])
    
    @property
    def beta(self):
        return 1/(KB*self.temperature)    
    
    @property
    def mn_args(self):
        return dict(zip(self.config[MN_ARGS + KEYS], self.config[MN_ARGS + VALUES]))
    
    @property
    def eq_args(self):
        return dict(zip(self.config[EQ_ARGS + KEYS], self.config[EQ_ARGS + VALUES]))
    
    @property
    def md_args(self):
        return dict(zip(self.config[MD_ARGS + KEYS], self.config[MD_ARGS + VALUES]))
    
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
        
        If only the last timestep is required, this approach is more effecient than
        building an entire list.
        
        Public API should only call the "timesteps" method.
        """
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        if len(timesteps):
            return Timestep(self.hdf[TIMESTEPS + "/" + str(max(timesteps))])
        else:
            return None
        
        
    def begin_timestep(self):
        """
        Creates a new empty current_timestep. If one currently exists, it is deleted.
        
        The _begin_timestep and _end_timestep logic are modeled after OpenGL's glBegin and glEnd. 
        the "current_timestep" link should only exist between calls 
        to begin_timestep and end_timestep. 
        
        The simulation may crash in between these calls to begin and end timestep, in this case, 
        there will be a partially completed current_timestep. For debugging purposes, the current_timestep
        may be loaded back into the system via _load_timestep. Note, in this case, current_timestep is likely not 
        complete, missing attributes will cause notifications via _load_timestep. 
        
        For development/debugging, there is no harm done in repeatedly calling begin_timestep, 
        only the end_timestep actually writes the current_timestep to timesteps[n+1]
        """
        # if there is a current timestep, we assume its garbage and delete it
        if self.hdf.id.links.exists(CURRENT_TIMESTEP):
            logging.warn("found previous \"current_timestep\" key, this means that end_timestep was not called.")
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
        move current_timestep to timesteps/n+1, and flush the file.
        
        This is the ONLY method that actually changes the completed timesteps. 
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
        """
        return the current_timestep.
        The current_timestep is created by begin_timestep, and flushed to the file and deleted
        by end_timestep. 
        
        The current_timestep should be treated as WRITE ONLY MEMORY!
        All read/write state variables should be instance variables of the System class, 
        such as univese, cg_postions, etc...
        
        It is very important to treat this as WRITE ONLY, reading from this var
        will completly screw up the logic of state evolution.
        
        """
        return Timestep(self.hdf[CURRENT_TIMESTEP])
    
    @property
    def dt(self):
        """
        The time step used in the coarse grained (Langevin) step. 
        specified in the config file.
        """
        return float(self.config[DT])
    
    @property
    def timesteps(self):
        """
        A list of Timestep objects which are the completed langevin steps.
        """
        # get all the completed timesteps, the keys are usually out of order, so
        # have to sort them
        timesteps = [int(k) for k in self.hdf[TIMESTEPS].keys()]
        timesteps.sort()
        return [Timestep(self.hdf[TIMESTEPS + "/" + str(ts)]) for ts in timesteps]
        

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
        """
        Run the simulation for as many timesteps as specified by the configuration.
        
        This will automatically start at the last completed timestep and proceed untill
        all the specified timesteps are completed.
        """
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
        """
        perform a forward euler step (the most idiotic and unstable of all possible integrators)
        to integrate the state variables. 
        
        reads self.cg_velocities (or possibly self.cg_forces) to calculat the diffusion matrix, 
        averages cg_forces, calculates D*f and notifies each subsystem with it's respective part
        of this vector to translate. 
        
        X[n+1] = X[n] + dt*dX/dt[n], and dX/dt is D*f.
        """
        
        # forward euler
        # result = coordinate_ops + beta() * dt * Df
        Df = dynamics.diffusion_force(self) 
        
        # Df is a column vector, change it to a row vector, as the subsystems
        # expect a length(ncg) row vector. 
        Df = Df.transpose()
        
        # per euler step, 
        # translate = dt * beta * dX/dt
        cg_translate = self.dt * self.beta * Df
        
        #write to ts
        self.current_timestep.cg_translate = cg_translate
        
        for i, s in enumerate(self.subsystems):
            s.translate(cg_translate[0,i*self.ncgs:i*self.ncgs+self.ncgs])
            
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
                               deffnm="eq", \
                               **self.eq_args)
    
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
                               deffnm="md", \
                               **self.md_args)
        
    def equilibriate(self):
        """
        Equilibriates (thermalizes) the structure stored in self.universe.
        
        takes the universe structure, inputs it to md, performs an equilibriation run, and
        read the result back into the universe. The equilibriated atomic state can then
        be used at the starting state the md sampling run, and for the langevin step.
        
        The MD is typically performed with position restraints on the protein. 
        
        The equilibriated position are also written to the current_timestep. 
        
        @precondition: self.universe contains an atomic state.
        @postcondition: self.universe now contains an equilbriated state.
            subsystems are notified.
            equililibriated state is written to current_timestep. 
        """
        logging.info("performing equilibriation")
        with self.setup_equilibriate() as eqsetup:
            mdres = md.run_md(eqsetup.dirname, **eqsetup)
            self.universe.load_new(mdres.structs[0])
        
        self.current_timestep.atomic_equilibriated_positions = self.universe.atoms.positions
        [s.equilibriated() for s in self.subsystems]
        
    def md(self):
        """
        Perform a set of molecular dynamics runs using the atomic state 
        of self.universe as the starting structure. The state of self.universe
        is NOT modified. 
        
        @precondition: self.universe contains a valid atomic state.
        @postcondition: self.cg_positions, self.cg_forces, self.cg_velocities[:] 
            are populated with statistics collected from the md runs.
        """
        with self.setup_md() as mdsetup:
            mdres = md.run_md(mdsetup.dirname, **mdsetup)            
            self._processes_trajectories(mdres.trajectories)
        
    def _processes_trajectories(self, trajectories):
        """
        reads each given atomic trajectory, extracts the
        coarse grained information, updates state variables with 
        this info, and saves it to the output file.
        
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
        timestep.flush()
            

    def topology_changed(self):
        """
        Notify the system that the topology of the universe was changed. 
        This causes the system to generate a new topology file, 
        and notify all subsystems that the universe was changed.
        """
        pass
    
    def solvate(self):
        """
        Not implemented yet.
        """
        pass
        
    def minimize(self):
        """
        Take the current structure (in self.universe) and minimize it via md.
        
        Loads the self.universe with the minimized structure and notifies all the 
        subsystems.
        
        @precondition: 
        self.universe contains the current atomic state
        
        @postcondition: 
        self.universe is loaded with the minimized structure
        all subsystems are notified.
        """
        logging.info("Performing minimization")
        with md.minimize(struct=self.universe, \
                         top=self.top, \
                         posres = self.posres, \
                         nsteps=self.config[MN_STEPS], \
                         deffnm="mn", \
                          **self.mn_args) as mn:
        
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
        
    def _load_timestep(self, ts):
        """
        Load the system - set the current timestep and the state variables to those 
        set in a Timestep object. 
        
        This method is usefull for debugging / development, so the state of the system
        can be set without performing an MD run. This way methods that change the state
        of the System, such as evolve() can be debugged. 
        """ 
        
        # check to see if the object can be used at a timestep object
        attrs = ["cg_positions", "cg_velocities", "cg_forces", "atomic_starting_positions"]
        ists = [hasattr(ts, attr) for attr in attrs].count(True) == len(attrs)
        
        if ists:      
            try:
                self.universe.atoms.positions[()] = ts.atomic_starting_positions
            except:
                print("failed to set atomic positions")
                
            try:
                self.cg_positions[()] = ts.cg_positions
            except:
                print("failed to set cg_positions")
                
            try:
                self.cg_velocities = ts.cg_velocities
            except:
                print("failed to set cg_velocities")
                
            try:
                self.cg_forces[()] = ts.cg_forces
            except:
                print("failed to set cg_forces")
            return
        
        # its not a timestep, so try as a integer
        try:
            return self._load_timestep(self.timesteps[int(ts)])
        except ValueError:
            raise ValueError("Assumed timestep was an index as it did not have required attributes, but could not convert to integer")

            
        
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
    "subsystem_selects": "not resname SOL",
    "cg_steps":5,
    "dt":10.0,
    "top_args": {},
    "mn_steps":5,
    "eq_steps":50,
    "md_steps":50,
    "multi":4,
    "solvate":False,
    } 

dpc = { 
    'box' : [90.0, 90.0, 90.0],      
    'temperature' : 300.0, 
    'struct': 'DPC-Self-CHARMM36.pdb',
    'top' :  '54_DPC_CHARMM36_h2o.top',
    "subsystem_selects": ["resname DPC"],
    "subsystem_args":["resid unique"],
    "cg_steps":5,
    "dt":0.1,
    "top_args": {},
    "mn_steps":5,
    "eq_steps":50,
    "md_steps":1000,
    "multi":4,
    "solvate":False,
    } 
    
def ctest(pdb, hdf):
    Au2['struct'] = pdb
    create_config(fid=hdf, **Au2)
    
def dpctest(hdf):
    create_config(fid=hdf, **dpc)

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
                  posres = None,
                  temperature = 300,
                  subsystem_factory = "dms2.subsystems.RigidSubsystemFactory",
                  subsystem_selects = ["not resname SOL"],
                  subsystem_args = [],
                  cg_steps = 10,
                  dt  = 0.1,
                  mn_steps = 500,
                  md_steps = 100,
                  multi = 1,
                  eq_steps = 10,
                  mn_args = {},
                  eq_args = DEFAULT_EQ_ARGS,
                  md_args = DEFAULT_MD_ARGS,
                  solvate = False,
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



        attr(CG_STEPS, int, cg_steps)
        attr(DT, float, dt)
        attr(TEMPERATURE, float, temperature)
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
        else:
            # use user specified top
            print("using user specified topology file {}".format(top))

            filedata_fromfile(TOPOL_TOP, top)
            filedata_fromfile(POSRES_ITP, POSRES_ITP)
            filedata_fromfile(STRUCT_PDB, struct)
            
        # try to make the subsystems.
         
        
        
        
        try:
            # make a fake 'System' object so we can test the subsystem factory.
            dummysys = None
            with tempfile.NamedTemporaryFile(suffix=".pdb") as f:  
                # EXTREMLY IMPORTANT to flush the file, 
                # NamedTemporaryFile returns an OPEN file, and the array writes to this file, 
                # so we need to flush it, as this file handle remains open. Then the MDAnalysis
                # universe opens another handle and reads its contents from it. 
                data = src_files[STRUCT_PDB][()]
                data.tofile(f.file)
                f.file.flush()
                universe = MDAnalysis.Universe(f.name)
                dummysys = collections.namedtuple('dummysys', 'universe')(universe)
                
            # we have a fake sys now, can call subsys factory
            factory = util.get_class(subsystem_factory)
            test_ncgs, test_ss = factory(dummysys, subsystem_selects, *subsystem_args)
            
            print("subsystem factory appears to work, produces {} cg variables for each {} subsystems.".format(test_ncgs, len(test_ss)))
            
            conf[SUBSYSTEM_FACTORY] = subsystem_factory
            conf[SUBSYSTEM_SELECTS] = subsystem_selects
            conf[SUBSYSTEM_ARGS] = subsystem_args
            print("{}: {}".format(SUBSYSTEM_FACTORY, subsystem_factory))

        except Exception, e:
            print("error creating subsystem_class, {}".format(e))
            raise e

        hdf.create_group("timesteps")
        print("creation of simulation file {} complete".format(fid))
        



        


    

