import dms2.subsystems.legendre_subsystem as ss
from sys import argv
import MDAnalysis as md
import numpy as np

def Loop(U, nframes, SS, vel, global_nframes, sel):
    for ts in xrange(global_nframes,nframes):
        U.trajectory.next()
        global_nframes += 1
        vel += eval('U.atoms.{}'.format(sel))

    vel_avg = vel / global_nframes
    Pi = SS.ComputeCG_Vel(vel_avg)

    return np.reshape(Pi,(1,3)), global_nframes, U

def Plot_Time_Integral(pdb, traj, kmax, sel, delta_frames, ofname):
    print 'Reading input ...'
    U = md.Universe(pdb,traj)
    nframes = U.trajectory.numframes

    trials = nframes / delta_frames
    
    _, SS = ss.LegendreSubsystemFactory(U, ['all'], kmax)
    SS = SS[0]
    SS.universe_changed(U)
    SS.equilibriated()

    vel = np.zeros((U.atoms.numberOfAtoms(),3))
    global_nframes = 0
    
    print 'Computing the time integral for a total of {} md runs'.format(trials)
    
    for i in xrange(trials):
        fp = open(ofname,'a')
        integral, global_nframes, U = Loop(U, (i+1)*(nframes/nframes/trials - (trials % trials)), SS, vel, global_nframes, sel)
        #print SS.ComputeCG(coords)[3], SS.ComputeCG(U.atoms.positions)[3]
        np.savetxt(fp, integral)
        #print global_nframes
        fp.close()
        
    print 'Done! Output successfully written to {}'.format(ofname)