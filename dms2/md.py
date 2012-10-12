'''
Created on Aug 8, 2012

@author: andy
'''
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

class ResourceManager(object):
    def __init__(self, files):
        import types
        self._files = files if isinstance(files, types.ListType) else [files]
        
    def  __enter__(self):
        return self
    
    def __exit__(self, tp, value, traceback):
        for f in self._files:
            try:
                #os.remove(f)
                print("would have removed {}".format(f))
            except OSError:
                logging.warn('could not remove {}'.format(f))
    
    @property
    def files(self):
        return self._files
    
def test_md(config, atoms):
#    return ResourceManager("dpc_125_nvt.trr")

    files = ["nvt60.trr", "nvt61.trr", "nvt62.trr", "nvt63.trr", "nvt64.trr", 
             "nvt65.trr", "nvt66.trr", "nvt67.trr", "nvt68.trr", "nvt69.trr"]
    
    return ResourceManager(files)
    
    
def md(struct, top):
    """
    performs a molecular dynamics run
    @return: a tuple of (coordinate_file_names, velocity_file_names, force_file_names)
    """
    w = MDAnalysis.Writer("md_in.pdb")
    w.write(atoms)
    w.close()
    
    g=gromacs.grompp(f=config["md_mdp"], c="md_in.pdb", p=config["top"], o="md.tpr")

    if os.environ.has_key("SLURM_JOBID"):
        logging.info("running in SLURM")
        m=gromacs.run.MDrunnerOpenMPI(deffnm="md")
    else:
        logging.info("not using SLURM")
        m=MDrunnerLocal(deffnm="md")
    
    m.run()
    
    os.remove("md_in.pdb")
    os.remove("md.tpr")
    
    return ResourceManager("md.trr")

    
def minimize(struct, top, minimize_dir='em', minimize_mdp=gromacs.config.templates['em.mdp'], 
             minimize_output='em.pdb', minimize_deffnm="em", mdrunner=None, **kwargs):
    """Energy minimize the system.

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
    return gromacs.setup.energy_minimize(dirname=minimize_dir, mdp=minimize_mdp, struct=struct, 
                                         top=top, output=minimize_output, deffnm=minimize_deffnm, 
                                         mdrunner=mdrunner, **kwargs)
    
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
    kwargs.setdefault('struct', 'em/em.pdb')
    kwargs.setdefault('qname', 'PR_GMX')
    kwargs.setdefault('define', '-DPOSRES')
    
    # reduce size of output files
    #kwargs.setdefault('nstxout', '50000')   # trr pos
    #kwargs.setdefault('nstvout', '50000')   # trr veloc
    #kwargs.setdefault('nstfout', '0')       # trr forces
    #kwargs.setdefault('nstlog', '500')      # log file
    #kwargs.setdefault('nstenergy', '2500')  # edr energy
    #kwargs.setdefault('nstxtcout', '5000')  # xtc pos
    kwargs.setdefault('deffnm','md')
    kwargs.setdefault('mdp',config.templates['md_CHARMM27.mdp'])
    
    return gromacs.setup._setup_MD(dirname, **kwargs)
    
def equilibriate_restrained(universe):
    pass

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
    
    @return a dict with the following keys: {"top", "struct", "posres"}, where
    the values are the file names of the resulting topology, structure, and position restraint files.
    """
    if top is None:
        logging.info("config did not specify a topology, autogenerating using pdb2gmx...")
        pdb2gmx_args = {"ff":ff, "water":water, "ignh":ignh}
        pdb2gmx_args.update(top_args)
        result = gromacs.setup.topology(struct, protein, "system.top", dirname, **pdb2gmx_args)
        result["posres"] = protein + "_posres.itp"
    else:
        logging.info("config specified a topology, \{\"top\":{}, \"struct\":{}\}".format(top, struct))
        result={"top":top, "struct":struct, "posres":posres}
    return result
        
    
def solvate(struct, top,
            distance=0.9, boxtype='cubic',
            concentration=0, cation='NA+', anion='CL-',
            water='spc', solvent_name='SOL', with_membrane=False,
            ndx = 'main.ndx', mainselection = '"Protein"',
            solvate_dir='solvate',
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
    return gromacs.setup.solvate(struct, top,
            distance, boxtype,
            concentration, cation, anion,
            water, solvent_name, with_membrane,
            ndx, mainselection,
            solvate_dir,  **kwargs)
    
   

def rm(fname):
    try:
        os.remove(fname)
    except OSError:
        logging.warn('could not remove {}'.format(fname))
        
        


    

def run_namd(conf_file):
    logging.debug('starting run_namd(conf_file={})'.format(conf_file))
    max_retry = 5
    which_namd=which('namd2')
    
    #determine MPI command
    cmd = ['mpirun']
    
    #MPI if running on load leveler looks like:
    #mpirun -np $LOADL_TOTAL_TASKS -machinefile $LOADL_HOSTFILE $NAMD2
    LOADL_TOTAL_TASKS=os.environ.get('LOADL_TOTAL_TASKS', None)
    LOADL_HOSTFILE=os.environ.get('LOADL_HOSTFILE',None)
    
    if(LOADL_TOTAL_TASKS and LOADL_HOSTFILE):
        logging.debug('running in load leveler')
        cmd.append('-np')
        cmd.append(LOADL_TOTAL_TASKS)
        cmd.append('-machinefile')
        cmd.append(LOADL_HOSTFILE)
        
    cmd.append(which_namd)
    cmd.append(conf_file)
    
    out_open = lambda mode: open('namd_out.tmp.log', mode)
    err_open = lambda mode: open('namd_err.tmp.log', mode)
    
    for retry in range(0,max_retry):
        logging.info('calling namd retry {} with cmd: {}'.format(retry, cmd))
    
        proc = subprocess.Popen(args=cmd, stdout=out_open('w'), stderr=err_open('w'))
        code=proc.wait()

        # concatenate tmp logs with main logs
        shutil.copyfileobj(out_open('r'), open('namd_out.log', 'w+'))
        shutil.copyfileobj(err_open('r'), open('namd_err.log', 'w+'))
    
        # search output for bits. Note, this is really ugly, and needs to be cleaned up
        fatal_re = re.compile("FATAL ERROR")
        fatal_error = len([True for line in out_open('r').readlines() if fatal_re.search(line)]) > 0
        if fatal_error:
            logging.error("'FATAL ERROR' found in namd output file, see log files namd_out.log and namd_err.log")
            continue
        
        end_re = re.compile("End of program")
        end_of_program = len([True for line in out_open('r').readlines() if end_re.search(line)]) > 0
        if not end_of_program:
            logging.error("'End of program' NOT found in namd output, see log files namd_out.log and namd_err.log")
            continue
        
        # reached here, no error, can exit
        logging.debug('finished run_namd(...) = {}'.format(code))
        return code
    
    #after retry loop, max retries exceeded
    logging.critical("Maximum namd retries exceeded, see namd output logs, terminating...")
    epic_fail()
    
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
        
def which(program):
    """
    determine the absolute path for a program,
    python version of the unix 'which' command
    """
    for path in os.environ.get('PATH', '').split(':'):
        if os.path.exists(os.path.join(path, program)) and \
           not os.path.isdir(os.path.join(path, program)):
            return os.path.join(path, program)
    return None
