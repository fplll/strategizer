# -*- coding: utf-8 -*-

u"""
Machine dependent constants.

.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""
from __future__ import absolute_import
from .config import logging
import pickle

logger = logging.getLogger(__name__)


def load_mdc():
    global nps
    with open("mdc.data", "rb") as f:
        nps = float(pickle.load(f))
    logger.info("enum nodes per sec: %13.1f", nps)

load_mdc()


def nodes_per_sec(n):
    global nps
    return nps
