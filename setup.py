# setuptools installation of GromacsWrapper
# Copyright (c) 2008-2011 Oliver Beckstein <orbeckst@gmail.com>
# Released under the GNU Public License 3 (or higher, your choice)
#
# See the files INSTALL and README for details or visit
# http://sbcb.bioch.ox.ac.uk/oliver/software/GromacsWrapper/
from __future__ import with_statement

from ez_setup import use_setuptools
use_setuptools()
from setuptools import setup, find_packages

with open("README") as readme:
    long_description = readme.read()


version = "0.0.1"

setup(name="DMS",
      version=version,
      description="A python wrapper around the gromacs tools.",
      long_description=long_description,
      author="Oliver Beckstein",
      author_email="orbeckst@gmail.com",
      license="GPLv3",
      url="://github.com/orbeckst/GromacsWrapper",
      download_url="https://github.com/orbeckst/GromacsWrapper/downloads",
      keywords="science Gromacs analysis 'molecular dynamics'",
      classifiers=['Development Status :: 4 - Beta',
                   'Environment :: Console',
                   'Intended Audience :: Science/Research',
                   'License :: OSI Approved :: GNU General Public License (GPL)',
                   'Operating System :: POSIX',
                   'Operating System :: MacOS :: MacOS X',
                   'Programming Language :: Python',
                   'Topic :: Scientific/Engineering :: Bio-Informatics',
                   'Topic :: Scientific/Engineering :: Chemistry',
                   'Topic :: Software Development :: Libraries :: Python Modules',
                   ],
      packages=find_packages(exclude=['tests','scripts','extras','doc/examples']),
      scripts = [],
      package_data={'dms2': ['templates/*.sge', 'templates/*.pbs',  # template files
                                'templates/*.ll', 'templates/*.sh',
                                'templates/*.mdp', 'templates/*.cfg',
                                'external/GridMAT-MD_v1.0.2/GridMAT-MD.pl']   # external bundled scripts
                    },
      install_requires = ['numpy>=1.0',
                          'scipy',        # numkit needs it
                          ],              # basic package (w/o analysis)
      extras_require = {
                'numkit': ['scipy'],
                },
      zip_safe = True,
)


