# -*- coding: utf-8 -*-
u"""
.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""

import pickle
import argparse
from fpylll import FPLLL
from fpylll.tools.benchmark import bench_enumeration
from strategizer.config import logging
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description='options')
parser.add_argument('-e', '--enumthreads', help='number of fplll threads to use', type=int, default=-1)
args = parser.parse_args()

FPLLL.set_threads(args.enumthreads)

time = 0
dim = 40
while time < 10:
    dim += 2
    nodes, time = bench_enumeration(dim)
    logger.info("  fplll :: dim: %i, nodes: %12.1f, time: %6.4fs, nodes/s: %12.1f"%(dim, nodes, time, nodes/time))

f = open("mdc.data", "wb")
pickle.dump(nodes/time, f)
f.close()
