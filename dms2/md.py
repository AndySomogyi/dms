"""
Functions for setting up and performing md calculations.

All function arguments which refer to file input may be either a 
string, in which they are interpreted as a file path, or a numpy array, in 
which they are interpreted as data buffers containing the file bytes.

Created on Aug 8, 2012
@author: andy
"""
import os.path
import os
import time
import shutil
import logging
import re
import subprocess
import MDAnalysis
import gromacs.setup
import gromacs.run
import config
from collections import namedtuple
from os import path
from util import data_tofile
import shutil

from collections import Mapping, Hashable 


class MDManager(Mapping, Hashable): 
    __slots__ = ("__dict", "dirname")

    def __init__(self, *args, **kwargs): 
        print("__init__")
        self.__dict = dict(*args, **kwargs)
        print(self.__dict) 
        
        if not self.__dict.has_key("dirname"):
            raise ValueError("MDManager arguments must contain a \"dirname\" key")
        
        self.dirname = self.__dict["dirname"]
        del self.__dict["dirname"]
        
        if not os.path.isdir(self.dirname):
            raise IOError("dirname of {} is not a directory".format(self.dirname))
        
        # save the abs path, user could very likely change directories.
        self.dirname = os.path.abspath(self.dirname)

    def __len__(self): 
        return len(self.__dict) 

    def __iter__(self): 
        return iter(self.__dict) 

    def __getitem__(self, key): 
        return self.__dict[key] 

    def __hash__(self): 
        return hash(frozenset(self.__dict.iteritems())) 

    def __repr__(self): 
        return "MDManager({})".format(self.__dict) 
    
    def __enter__(self):
        print("__enter__")
        return self
    
    def __exit__(self, *args):
        print("deleting {}".format(self.dirname))
        #shutil.rmtree(self.dirname)

        
        
def test(dirname):
    return MDManager({'dirname':dirname})



class MDrunnerLocal(gromacs.run.MDrunner):
    """Manage running :program:`mdrun` as mpich2_ multiprocessor job with the SMPD mechanism.

    .. _mpich2: http://www.mcs.anl.gov/research/projects/mpich2/
    """
    mdrun = "mdrun"
    mpiexec = "mpiexec"

    def mpicommand(self, *args, **kwargs):
        """Return a list of the mpi command portion of the commandline.

        Only allows primitive mpi at the moment:
           *mpiexec* -n *ncores* *mdrun* *mdrun-args*

        (This is a primitive example for OpenMP. Override it for more
        complicated cases.)
        """

        return ["mpiexec", "-n", "4"]


    
def test_md(config, atoms):
#    return ResourceManager("dpc_125_nvt.trr")

    files = ["nvt60.trr", "nvt61.trr", "nvt62.trr", "nvt63.trr", "nvt64.trr", 
             "nvt65.trr", "nvt66.trr", "nvt67.trr", "nvt68.trr", "nvt69.trr"]
    
    return ResourceManager(files)
    
    
def MD(dirname='MD', **kwargs):
    """Set up, but don't actually run an equilibrium MD.

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values. Note that
    setting *mainselection* = ``None`` will disable many of the automated
    choices and is often recommended when using your own mdp file.

    :Keywords:
       *dirname*
          set up under directory dirname [MD]
       *struct*
          input structure (gro, pdb, ...) [MD_POSRES/md_posres.pdb]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/md.mdp]
       *ndx*
          index file (supply when using a custom mdp)
       *includes*
          additional directories to search for itp files
       *mainselection*
          ``make_ndx`` selection to select main group ["Protein"]
          (If ``None`` then no canonical index file is generated and
          it is the user's responsibility to set *tc_grps*,
          *tau_t*, and *ref_t* as keyword arguments, or provide the mdp template
          with all parameter pre-set in *mdp* and probably also your own *ndx*
          index file.)
       *deffnm*
          default filename for Gromacs run [md]
       *runtime*
          total length of the simulation in ps [1000]
       *dt*
          integration time step in ps [0.002]
       *qscript*
          script to submit to the queuing system; by default
          uses the template :data:`gromacs.config.qscript_template`, which can
          be manually set to another template from :data:`gromacs.config.templates`;
          can also be a list of template names.
       *qname*
          name to be used for the job in the queuing system [MD_GMX]
       *mdrun_opts*
          option flags for the :program:`mdrun` command in the queuing system
          scripts such as "-stepout 100 -dgdl". [""]
       *kwargs*
          remaining key/value pairs that should be changed in the template mdp
          file, e.g. ``nstxtcout=250, nstfout=250`` or command line options for
          :program`grompp` such as ``maxwarn=1``.

    :Returns: a dict that can be fed into :func:`gromacs.setup.MD`
              (but check, just in case, especially if you want to
              change the *define* parameter in the mdp file)    
    """

    logging.info("[%(dirname)s] Setting up MD..." % vars())
    
    # set the interval to save output values.
    # nstout is NOT a valid mdp name, however we need all the values (force, vel, pos) to
    # be written at at the same time interval, so I came up with this key to set the
    # output interval for all of them.
    nstout = MD.defaults["nstout"]
    if kwargs.has_key("nstout"):
        nstout = kwargs.pop("nstout")
    kwargs.setdefault('nstxout', nstout)   # trr pos
    kwargs.setdefault('nstvout', nstout)   # trr veloc
    kwargs.setdefault('nstfout', nstout)   # trr forces
    
    kwargs.setdefault('mdp',config.templates['md_CHARMM27.mdp'])
    
    return gromacs.setup._setup_MD(dirname, **kwargs)


MD.defaults = {
               "nstout":10,          # output things every so many steps.
               'mdp':config.templates['md_CHARMM27.mdp']
               }

def MD_config_get(conf, what):
    """
    get a value from the config, if it does not have it, use the
    appropriate defaults. 
    
    This is the same value that MD will use.
    """
    result = None
    if conf.has_key(what):
        result = conf[what]
    else:
        result = MD.defaults[what]
    return result

    
def minimize(struct, top, dirname='em', minimize_mdp=config.templates['em.mdp'], 
             minimize_output='em.pdb', minimize_deffnm="em", mdrunner=MDrunnerLocal, **kwargs):
    """
    Energy minimize a system.
    
    Creates the directory minimize_dir, and all operations are performed there. 
    
    @param struct: name of structure file
    @param top: name of top file
    
    @return: a dictionary with the following values:
        'struct': final_struct,
        'top': topology,
        'mainselection': mainselection,
        'dirname': dir where minimization took place.

    This sets up the system (creates run input files) and also runs
    ``mdrun_d``. Thus it can take a while.

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values.

    :Keywords:
       *dirname*
          set up under directory dirname [em]
       *struct*
          input structure (gro, pdb, ...) [solvate/ionized.gro]
       *output*
          output structure (will be put under dirname) [em.pdb]
       *deffnm*
          default name for mdrun-related files [em]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/em.mdp]
       *includes*
          additional directories to search for itp files
       *mdrunner*
          :class:`gromacs.run.MDrunner` class; by defauly we
          just try :func:`gromacs.mdrun_d` and :func:`gromacs.mdrun` but a
          MDrunner class gives the user the ability to run mpi jobs
          etc. [None]
       *kwargs*
          remaining key/value pairs that should be changed in the
          template mdp file, eg ``nstxtcout=250, nstfout=250``.

    .. note:: If :func:`~gromacs.mdrun_d` is not found, the function
              falls back to :func:`~gromacs.mdrun` instead.
    """
    
    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    top = data_tofile(top, "src.top", dirname=dirname)
    
    # gromacs.setup.energy_minimize returns
    # { 'struct': final_struct,
    #   'top': topology,
    #   'mainselection': mainselection,
    # }
    result = gromacs.setup.energy_minimize(dirname=dirname, mdp=minimize_mdp, struct=struct, 
                                         top=top, output=minimize_output, deffnm=minimize_deffnm, 
                                         mdrunner=mdrunner, **kwargs)
    result["dirname"] = dirname
    return MDManager(result)
    
def MD_restrained(dirname='MD_POSRES', **kwargs):
    """Set up MD with position restraints.

    Additional itp files should be in the same directory as the top file.

    Many of the keyword arguments below already have sensible values. Note that
    setting *mainselection* = ``None`` will disable many of the automated
    choices and is often recommended when using your own mdp file.

    :Keywords:
       *dirname*
          set up under directory dirname [MD_POSRES]
       *struct*
          input structure (gro, pdb, ...) [em/em.pdb]
       *top*
          topology file [top/system.top]
       *mdp*
          mdp file (or use the template) [templates/md.mdp]
       *ndx*
          index file (supply when using a custom mdp)
       *includes*
          additional directories to search for itp files
       *mainselection*
          :program:`make_ndx` selection to select main group ["Protein"]
          (If ``None`` then no canonical index file is generated and
          it is the user's responsibility to set *tc_grps*,
          *tau_t*, and *ref_t* as keyword arguments, or provide the mdp template
          with all parameter pre-set in *mdp* and probably also your own *ndx*
          index file.)
       *deffnm*
          default filename for Gromacs run [md]
       *runtime*
          total length of the simulation in ps [1000]
       *dt*
          integration time step in ps [0.002]
       *qscript*
          script to submit to the queuing system; by default
          uses the template :data:`gromacs.config.qscript_template`, which can
          be manually set to another template from :data:`gromacs.config.templates`;
          can also be a list of template names.
       *qname*
          name to be used for the job in the queuing system [PR_GMX]
       *mdrun_opts*
          option flags for the :program:`mdrun` command in the queuing system
          scripts such as "-stepout 100". [""]
       *kwargs*
          remaining key/value pairs that should be changed in the template mdp
          file, eg ``nstxtcout=250, nstfout=250`` or command line options for
          ``grompp` such as ``maxwarn=1``.

          In particular one can also set **define** and activate
          whichever position restraints have been coded into the itp
          and top file. For instance one could have

             *define* = "-DPOSRES_MainChain -DPOSRES_LIGAND"

          if these preprocessor constructs exist. Note that there
          **must not be any space between "-D" and the value.**

          By default *define* is set to "-DPOSRES".

    :Returns: a dict that can be fed into :func:`gromacs.setup.MD`
              (but check, just in case, especially if you want to
              change the ``define`` parameter in the mdp file)

    .. Note:: The output frequency is drastically reduced for position
              restraint runs by default. Set the corresponding ``nst*``
              variables if you require more output.
    """
    logging.info("[%(dirname)s] Setting up MD with position restraints..." % vars())
    kwargs.setdefault('qname', None)
    kwargs.setdefault('define', '-DPOSRES')
    
    # reduce size of output files
    #kwargs.setdefault('nstxout', '50000')   # trr pos
    #kwargs.setdefault('nstvout', '50000')   # trr veloc
    #kwargs.setdefault('nstfout', '0')       # trr forces
    #kwargs.setdefault('nstlog', '500')      # log file
    #kwargs.setdefault('nstenergy', '2500')  # edr energy
    #kwargs.setdefault('nstxtcout', '5000')  # xtc pos
    kwargs.setdefault('mdp',config.templates['md_CHARMM27.mdp'])
    
    return MDManager(gromacs.setup._setup_MD(dirname, **kwargs))
    

md_defaults = {
    #title       : Protein-ligand complex NVT equilibration 
    "define"      : "-DPOSRES",     # position restrain the protein and ligand
    #; Run parameters
    "integrator"  : "md",           # leap-frog integrator
    "nsteps"      : 50000,          # 2 * 50000 = 100 ps
    "dt"          : 0.002,          # 2 fs
    #; Output control
    "nstxout"     : 100,            # save coordinates every 0.2 ps
    "nstvout"     : 100,            # save velocities every 0.2 ps
    "nstenergy"   : 100,            # save energies every 0.2 ps
    "nstlog"      : 100,            # update log file every 0.2 ps
    #"energygrps"  : "Protein JZ4",  #
    #; Bond parameters
    "constraint_algorithm" : "lincs",   # holonomic constraints 
    "constraints"     : "all-bonds",    # all bonds (even heavy atom-H bonds) constrained
    "lincs_iter"      : 1,          # accuracy of LINCS
    "lincs_order"     : 4,          # also related to accuracy
    #; Neighborsearching
    "ns_type"     : "grid",         # search neighboring grid cells
    "nstlist"     : 5,              # 10 fs
    "rlist"       : 0.9,            # short-range neighborlist cutoff (in nm)
    "rcoulomb"    : 0.9,            # short-range electrostatic cutoff (in nm)
    "rvdw"        : 1.4,            # short-range van der Waals cutoff (in nm)
    #; Electrostatics
    "coulombtype"     : "PME",      # Particle Mesh Ewald for long-range electrostatics
    "pme_order"       : 4,          # cubic interpolation
    "fourierspacing"  : 0.16,       # grid spacing for FFT
    #; Temperature coupling
    "tcoupl"      : "Berendsen",    # Berendsen thermostat
    "tc-grps"     : "Protein SOL",  # two coupling groups - more accurate
    "tau_t"       : [0.1, 0.1],    # time constant, in ps
    "ref_t"       : [300, 300],    # reference temperature, one for each group, in K
    #; Pressure coupling
    "pcoupl"      : "no",           # no pressure coupling in NVT
    #; Periodic boundary conditions
    "pbc"         : "xyz",          # 3-D PBC
    #; Dispersion correction
    "DispCorr"    : "EnerPres",     # account for cut-off vdW scheme
    #; Velocity generation
    "gen_vel"     : "yes",          # assign velocities from Maxwell distribution
    "gen_temp"    : 300,            # temperature for Maxwell distribution
    "gen_seed"    : -1,             # generate a random seed
    }



def topology(struct, protein="protein", top=None, dirname="top", posres=None, ff="charmm27", water="spc", ignh=True, **top_args):
    """
    Generate a topology for a given structure.
    
    @return a dict with the following keys: {"top", "struct", "posres", "dirname"}, where
    the values are the file names of the resulting topology, structure, and position restraint files.
    """
    if top is None:
        logging.info("config did not specify a topology, autogenerating using pdb2gmx...")
        pdb2gmx_args = {"ff":ff, "water":water, "ignh":ignh}
        pdb2gmx_args.update(top_args)
        struct = data_tofile(struct, "src.pdb", dirname=dirname)
        result = gromacs.setup.topology(struct, protein, "system.top", dirname, **pdb2gmx_args)
        result["posres"] = path.abspath(path.join(dirname, protein + "_posres.itp"))
        result["dirname"] = dirname
    else:
        logging.info("config specified a topology, \{\"top\":{}, \"struct\":{}\}".format(top, struct))
        result={"top":top, "struct":struct, "posres":posres, "dirname":None}
    
    return MDManager(result)
        
    
def solvate(struct, top, box,
            concentration=0, cation='NA', anion='CL',
            water='spc', solvent_name='SOL', with_membrane=False,
            ndx = 'main.ndx', mainselection = '"Protein"',
            dirname='solvate',
            **kwargs):
    """Put protein into box, add water, add counter-ions.

    Currently this really only supports solutes in water. If you need
    to embedd a protein in a membrane then you will require more
    sophisticated approaches.

    However, you *can* supply a protein already inserted in a
    bilayer. In this case you will probably want to set *distance* =
    ``None`` and also enable *with_membrane* = ``True`` (using extra
    big vdw radii for typical lipids).

    .. Note:: The defaults are suitable for solvating a globular
       protein in a fairly tight (increase *distance*!) dodecahedral
       box.

    :Arguments:
      *struct* : filename
          pdb or gro input structure
      *top* : filename
          Gromacs topology
      *distance* : float
          When solvating with water, make the box big enough so that
          at least *distance* nm water are between the solute *struct*
          and the box boundary.
          Set *boxtype*  to ``None`` in order to use a box size in the input
          file (gro or pdb).
      *boxtype* : string
          Any of the box types supported by :class:`~gromacs.tools.Editconf`
          (triclinic, cubic, dodecahedron, octahedron). Set the box dimensions
          either with *distance* or the *box* and *angle* keywords.

          If set to ``None`` it will ignore *distance* and use the box
          inside the *struct* file.
      *box*
          List of three box lengths [A,B,C] that are used by :class:`~gromacs.tools.Editconf`
          in combination with *boxtype* (``bt`` in :program:`editconf`) and *angles*.
          Setting *box* overrides *distance*.
      *angles*
          List of three angles (only necessary for triclinic boxes).
      *concentration* : float
          Concentration of the free ions in mol/l. Note that counter
          ions are added in excess of this concentration.
      *cation* and *anion* : string
          Molecule names of the ions. This depends on the chosen force field.
      *water* : string
          Name of the water model; one of "spc", "spce", "tip3p",
          "tip4p". This should be appropriate for the chosen force
          field. If an alternative solvent is required, simply supply the path to a box
          with solvent molecules (used by :func:`~gromacs.genbox`'s  *cs* argument)
          and also supply the molecule name via *solvent_name*.
      *solvent_name*
          Name of the molecules that make up the solvent (as set in the itp/top).
          Typically needs to be changed when using non-standard/non-water solvents.
          ["SOL"]
      *with_membrane* : bool
           ``True``: use special ``vdwradii.dat`` with 0.1 nm-increased radii on
           lipids. Default is ``False``.
      *ndx* : filename
          How to name the index file that is produced by this function.
      *mainselection* : string
          A string that is fed to :class:`~gromacs.tools.Make_ndx` and
          which should select the solute.
      *dirname* : directory name
          Name of the directory in which all files for the solvation stage are stored.
      *includes*
          List of additional directories to add to the mdp include path
      *kwargs*
          Additional arguments are passed on to
          :class:`~gromacs.tools.Editconf` or are interpreted as parameters to be
          changed in the mdp file.

    """
    struct = data_tofile(struct, "src.pdb", dirname=dirname)
    top = data_tofile(top, "src.top", dirname=dirname)
    result = gromacs.setup.solvate(struct, top,
            1.0, "cubic", 
            concentration, cation, anion,
            water, solvent_name, with_membrane,
            ndx, mainselection,
            dirname, box="{} {} {}".format(*box))
    result["dirname"] = dirname
    result["top"] = top
    
    return MDManager(result)
    
    
def run_MD(dirname, md_runner=gromacs.run.MDrunner, **kwargs):
    """
    actually perform the md run.
    
    does not alter class state, only the file system is changed
    
    @param kwargs: a dictionary of arguments that are passed to mdrun.
    
                   mdrun, with -multi multiple systems are simulated in parallel. As many input files are 
                   required as the number of systems. The system number is appended to the run input and 
                   each output filename, for instance topol.tpr becomes topol0.tpr, topol1.tpr etc. 
                   The number of nodes per system is the total number of nodes divided by the number of systems.
                   
                   run_MD automatically creates n copies of the source tpr specified by deffnm.
                   
    @return: a named tuple, currently, this contains a list of resulting trajectgories in
        in result.trajectories.
                    
    """
    Result = namedtuple('Result', 'trajectories')
    
    # pick out the relevant mdrun keywords from kwargs
    mdrun_args = ["h","version","nice","deffnm","xvg","pd","dd","nt","npme","ddorder",
                  "ddcheck","rdd","rcon","dlb","dds","gcom","v","compact","seppot",
                  "pforce","reprod","cpt","cpnum","append","maxh","multi","replex",
                  "reseed","ionize"]
    keys = [i for i in kwargs.keys() if i in mdrun_args]
    kwargs = dict((i, kwargs[i]) for i in keys)
    ncores = 4
    
    multi = kwargs.pop("multi", 1)
    if multi > 1:
        kwargs["multi"] = multi
        files = map(lambda i: 
                    path.abspath(path.join(dirname, kwargs["deffnm"] + str(i) + ".trr")), 
                    range(multi))
        ncores = multi
    else:
        files = []
        
    runner = md_runner(dirname, **kwargs)
    runner.mpiexec = "mpiexec"
    runner.run_check(ncores = ncores)
    
    return Result(files)

def multi_files(dirname, deffnm, multi):
    """
    creates a list of filenames for a multi md simulation. 
    
    @param dirname: directory where the simulation was performed
    @param deffnm: default name of files
    @param multi: how many multi simulations were performed.
    """ 
    
    files = map(lambda i: 
                    path.abspath(path.join(dirname, deffnm + str(i) + ".trr")), 
                    range(multi))
    return files
        
        
    
def epic_fail():
    """
    determine if we are running in load leveler or slurm, and
    kill the job
    """
    
    # find out what job scheduler we are runing under
    if os.environ.get('LOADL_STEP_ID', None):
        cmd = "llcancel {}".format(os.environ.get('LOADL_STEP_ID'))
        logging.critical("running in load leveler, about to terminate with {}".format(cmd))
        os.system(cmd)
    elif os.environ.get('SLURM_JOB_ID', None):
        cmd = "scancel {}".format(os.environ.get('SLURM_JOB_ID'))
        logging.critical("running in slurm, about to terminate with {}".format(cmd))
        os.system(cmd)
    elif os.environ.get('PBS_JOBID', None):
        cmd = "qdel {}".format(os.environ.get('PBS_JOBID'))
        logging.critical("running in Torque / PBS, about to terminate with {}".format(cmd))
        os.system(cmd)
    else:
        logging.critical("running in unknown environment, now what the fuck do we do?")
    
    # now just wait for job manager to kill us
    while True:
        time.sleep(5)
        logging.critical("waiting for job to terminate...")
        

