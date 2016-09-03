# -*- coding: utf-8 -*-
u"""
.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""
import subprocess
import logging


"Git Revision"

git_revision = []
cmds = [("git", "show", "-s", "--format=%cd", "HEAD", "--date=short"),
        ("git", "rev-parse", "--abbrev-ref", "HEAD"),
        ("git", "show", "-s", "--format=%h", "HEAD", "--date=short")]

for cmd in cmds:
    try:
        r = subprocess.check_output(cmd).rstrip()
        git_revision.append(r)
    except ValueError:
        pass

git_revision = "-".join(git_revision)


# Logging

logging.basicConfig(level=logging.DEBUG,
                    format='%(levelname)s:%(name)s:%(asctime)s: %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S %Z',
                    filename='%s.log'%git_revision,)

# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(name)s: %(message)s',))
logging.getLogger('').addHandler(console)
