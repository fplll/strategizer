# -*- coding: utf-8 -*-

u"""
Callback variants of BKZ, Params, Strategies

These variants call back to some coordinating process to gather pruning and preprocessing parameters.

.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>
"""

from __future__ import absolute_import
from fpylll.fplll.bkz_param import BKZParam
from fpylll.fplll.bkz import BKZ
from fpylll.algorithms.bkz2 import BKZReduction as BKZ2
from .volumes import gaussian_heuristic


class CallbackBKZParam(BKZParam):
    """
    BKZ Parameters with callback support.

    ..  note ::

        The default `BKZParam` class is tightly coupled to C++, hence we generalize it here slightly.
    """
    def __init__(self, strategies, **kwds):
        BKZParam.__init__(self, strategies=[], **kwds)
        self.callback_strategies = strategies

    @property
    def strategies(self):
        return self.callback_strategies


class CallbackStrategy(object):
    """
    A strategy which calls back to gather pruning and preprocessing parameters.
    """
    def __init__(self, block_size, connection, preprocessing_block_sizes=None, pruning_parameters=None):
        """
        :param block_size: block size
        :param connection: a connection object used to send/receive
            data
        :param preprocessing_block_sizes: preprocessing block sizes
            (if ``None`` then callbacks are used to establish these)
        :param pruning_parameters: pruning parameters (if ``None``
            then callbacks are used to establish these)
        """
        self.block_size = block_size
        self.connection = connection
        self._preprocessing_block_sizes = preprocessing_block_sizes
        self._pruning_parameters = pruning_parameters

    def callback(self, what, *args):
        """
        Call back.

        :param what:
        :returns:
        :rtype:

        """
        data = [what] + list(args)
        self.connection.send(data)
        r = self.connection.recv()
        return r

    def get_pruning(self, r, radius, stats, target_success_probability):
        """Return next pruning parameters.

        :param r: radii vector
        :param radius: target radius
        :param stats: stats to extract preprocessing costs

        """
        preproc_time = float(stats.current.parent.get("preprocessing")["cputime"])
        if self._pruning_parameters is None:
            self._pruning_parameters = self.callback("pruning",
                                                     tuple(r),
                                                     radius,
                                                     preproc_time,
                                                     target_success_probability)

        gh_radius = gaussian_heuristic(r)
        gh_factor = radius/gh_radius
        closest_dist = 2**80
        best = None
        for pruning in self._pruning_parameters:
            if abs(pruning.gh_factor - gh_factor) < closest_dist:
                best = pruning
                closest_dist = abs(pruning.gh_factor - gh_factor)
        assert(best is not None)
        return best

    def __getattr__(self, name):
        """Hack to enable callbacks for preprocessing parameters.

        :param name:

        """
        if name == "preprocessing_block_sizes":
            if self._preprocessing_block_sizes is None:
                self._preprocessing_block_sizes = self.callback("preproc")
            return tuple(self._preprocessing_block_sizes)
        else:
            raise AttributeError("'%s' object has no attribute '%s'"%(type(self), name))


class CallbackBKZ(BKZ2):
    """BKZ 2.0 interfacing with ``CallbackStrategy`` objects."""

    def get_pruning(self, kappa, block_size, param, stats):
        """
        Ask for pruning parameters from coordinating process.

        :param kappa:      index
        :param block_size: block size
        :param param:      access to strategy objects
        :param stats:      passed on

        """
        strategy = param.strategies[block_size]

        radius = self.M.get_r(kappa, kappa) * self.lll_obj.delta
        r = [self.M.get_r(i, i) for i in range(kappa, kappa+block_size)]
        gh_radius = gaussian_heuristic(r)
        if (param.flags & BKZ.GH_BND and block_size > 30):
            radius = min(radius, gh_radius * param.gh_factor)

        try:
            ret = radius, 0, strategy.get_pruning(tuple(r), radius, stats, param.min_success_probability)
        except TypeError:
            ret = BKZ2.get_pruning(self, kappa, block_size, param, stats)
        return ret
