import dms2.subsystems.legendre_subsystem as ss
import MDAnalysis as md
import numpy as np
import matplotlib.pylab as plt

def Loop(U, nframes, SS, vel, global_nframes, sel):
    for ts in xrange(global_nframes,nframes):
        U.trajectory.next()
        global_nframes += 1
        vel += eval('U.atoms.{}'.format(sel))

    vel_avg = vel / global_nframes
    Pi = SS.ComputeCG_Vel(vel_avg)

    return Pi, global_nframes, U

def Plot_Time_Integral(pdb, traj, kmax, sel, delta_frames, ofname, plot):
    print 'Reading input ...'
    U = md.Universe(pdb,traj)
    nframes = U.trajectory.numframes

    trials = nframes / delta_frames
    
    N_CG, SS = ss.LegendreSubsystemFactory(U, ['all'], kmax)
    N_CG /= 3
    SS = SS[0]
    SS.universe_changed(U)
    SS.equilibriated()

    vel = np.zeros((U.atoms.numberOfAtoms(),3))
    global_nframes = 0
    
    print 'Computing the time integral for a total of {} md runs'.format(trials)
    fp = open(ofname,'w')
    
    for i in xrange(trials):
        integral, global_nframes, U = Loop(U, (i+1)*(nframes/trials - (nframes % trials)), SS, vel, global_nframes, sel)
        np.savetxt(fp, integral)
    
    fp.close()
    
    if plot:
        integral = np.loadtxt(ofname)
        delta = np.arange(delta_frames,nframes,delta_frames)
        
        for order in xrange(N_CG):
            plt.plot(delta,integral[order:integral.shape[0]:N_CG,:])
            plt.show()
        
    print 'Done! Output successfully written to {}'.format(ofname)