# -*- coding: utf-8 -*-
u"""
.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""

import pickle
from fpylll.tools import bench_enumeration
from strategizer.config import logging
logger = logging.getLogger(__name__)

nodes, time = bench_enumeration(70)
logger.info("  fplll :: nodes: %12.1f, time: %6.4fs, nodes/s: %12.1f"%(nodes, time, nodes/time))

f = open("mdc.data", "wb")
pickle.dump(nodes/time, f)
f.close()
