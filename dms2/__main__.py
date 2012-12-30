'''
Created on Dec 30, 2012

@author: andy
'''

import system
import util
import sys

print("hello")

print(__name__)

system.test()

#print(util.get_class(sys.argv[1]))

if len(sys.argv) == 3:
    system.ctest(sys.argv[1], sys.argv[2])
else:
    #import subsystems
    s=system.System("test.hdf")
    s._begin_timestep()
    s.minimize()
