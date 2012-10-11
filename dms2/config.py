'''
Created on Oct 11, 2012

@author: andy
'''
from pkg_resources import resource_filename, resource_listdir  #@UnresolvedImport

def _generate_template_dict(dirname):
    """
    Generate a list of included files *and* extract them to a temp space.
    
    Templates have to be extracted from the egg because they are used
    by external code. All template filenames are stored in
    :data:`config.templates`.
    """
    return dict((resource_basename(fn), resource_filename(__name__, dirname+'/'+fn))
                for fn in resource_listdir(__name__, dirname)
                if not fn.endswith('~'))

def resource_basename(resource):
    """Last component of a resource (which always uses '/' as sep)."""
    if resource.endswith('/'):
        resource = resource[:-1]
    parts = resource.split('/')
    return parts[-1]
    

templates = _generate_template_dict('templates')
"""
*dms* comes with a number of templates for run input files
and queuing system scripts. They are provided as a convenience and
examples but **WITHOUT ANY GUARANTEE FOR CORRECTNESS OR SUITABILITY FOR
ANY PURPOSE**.

All template filenames are stored in
:data:`dms2.config.templates`. Templates have to be extracted from
the dms2 python egg file because they are used by external
code: find the actual file locations from this variable.

**Gromacs mdp templates**

These are supplied as examples and there is **NO GUARANTEE THAT THEY
PRODUCE SENSIBLE OUTPUT** --- check for yourself!  Note that only
existing parameter names can be modified with
:func:`gromacs.cbook.edit_mdp` at the moment; if in doubt add the
parameter with its gromacs default value (or empty values) and
modify later with :func:`~gromacs.cbook.edit_mdp`.

The safest bet is to use one of the ``mdout.mdp`` files produced by
:func:`gromacs.grompp` as a template as this mdp contains all
parameters that are legal in the current version of Gromacs.

**Queuing system templates**

The queing system scripts are highly specific and you will need to add your
own into :data:`gromacs.config.qscriptdir`.
See :mod:`gromacs.qsub` for the format and how these files are processed.
"""


def test():
    print(templates)
    
