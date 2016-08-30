# -*- coding: utf-8 -*-
"""
Strategizers are used to find strategies for BKZ reduction.
"""

from fpylll import BKZ
from .bkz import CallbackStrategy
from .mdc import nodes_per_sec
from .volumes import gaussian_heuristic, gh_margin
from fpylll import prune


class EmptyStrategizer(object):
    """
    The empty strategizer returns answers with ``no preprocessing`` and ``no pruning``.
    """

    name = "EmptyStrategy"
    min_block_size = 0
    Strategy = CallbackStrategy
    pruner_method = "hybrid"
    pruner_precision = 53

    def __init__(self, block_size, pruner_method="hybrid", pruner_precision=53):
        """

        :param block_size: block size to consider

        """
        self.block_size = block_size
        self.queries = []
        self.pruner_method = pruner_method
        self.pruner_precision = pruner_precision 


    def what(self, queries):
        """
        Decide which question is asked in ``inp``.
        """
        what = None
        data = []
        for q in queries:
            if q is not None:
                if what is None:
                    what = q[0]
                    data.append(q[1:])
                else:
                    if what != q[0]:
                        raise ValueError("Cannot handle inconsistent queries '%s' != '%s'"%(what, q[0]))
            else:
                data.append(None)

        return what, tuple(data)

    def __call__(self, queries):
        """Dispatch calls.

        :param queries: an array of queries

        """
        what, data = self.what(queries)
        self.queries.append(tuple(queries))
        if what == "preproc":
            return self.preproc(data)
        if what == "pruning":
            return self.pruning(data)
        if what is None:
            return [None]*len(data)
        else:
            raise ValueError("Query for '%s' not supported."%what)

    def preproc(self, data):
        """No preprocessing.

        :param data: tuple of (state, r_ii)

        """
        return tuple()

    def pruning(self, data):
        """No pruning.

        :param data: tuple of (state, r_ii, radius, preprocessing time)

        """
        return ((1., [1.0]*self.block_size, 1.),)


class SimplePreprocStrategizerTemplate(EmptyStrategizer):
    """
    """

    name = "SimplePreprocStrategy-(start, stop, step_size)"

    def preproc(self, inp):
        """
        Preprocess with one tour of 8,16,24,…,block_size-20-1
        """
        return range(self.start, self.block_size-self.stop, self.step_size)


def SimplePreprocStrategizerFactory(start, stop, step_size):
    """
    Create ``SimplePreprocStrategizer`` for blocks in ``range(start, block_size-stop, step_size)``
    """
    name = "SimplePreprocStrategy-(%d, %d, %d)"%(start, stop, step_size)
    return type("SimplePreprocStrategizer",
                (SimplePreprocStrategizerTemplate,),
                {"name": name, "start": start, "stop": stop, "step_size": step_size,
                 "min_block_size": start+stop+1})

SimplePreprocStrategizer16248 = SimplePreprocStrategizerFactory(16, 24, 8)


class OneTourPreprocStrategizerTemplate(EmptyStrategizer):
    """
    """

    name = "OnePreprocStrategy-block_size"

    def preproc(self, inp):
        """
        Preprocess with one tour of self.preprocessing_block_size
        """
        return [self.preprocessing_block_size]


def OneTourPreprocStrategizerFactory(block_size):
    """
    Create ``OneTourPreprocStrategizer`` for for ``block_size``
    """
    name = "OnePreprocStrategy-%d"%(block_size)
    return type("OneTourPreprocStrategizer",
                (OneTourPreprocStrategizerTemplate,),
                {"name": name, "preprocessing_block_size": block_size,
                 "min_block_size": block_size+1})


progressiveStep = 10
progressiveMin = 22

class ProgressivePreprocStrategizerTemplate(EmptyStrategizer):
    """
    """

    name = "ProgressivePreprocStrategy-block_size"

    def preproc(self, inp):
        """
        Preprocess with one tour of b for increasing  b <= self.preprocessing_block_size
        """
        L = [self.preprocessing_block_size]
        x = self.preprocessing_block_size - progressiveStep
        step = progressiveStep
        while x > progressiveMin:
            L = [x] + L
            step -= 2
            x -= step
            step = max(step, 4)
        return L


def ProgressivePreprocStrategizerFactory(block_size):
    """
    Create ``ProgressivePreprocStrategizer`` for for ``block_size``
    """
    name = "ProgressivePreprocStrategy-%d"%(block_size)
    return type("ProgressivePreprocStrategizer",
                (ProgressivePreprocStrategizerTemplate,),
                {"name": name, "preprocessing_block_size": block_size,
                 "min_block_size": block_size+1})




class PruningStrategizer(EmptyStrategizer):
    """
    Minimise pruning parameters
    """

    name = "PruningStrategy"
    Strategy = CallbackStrategy
    GH_FACTORS_STEPS=10

    def pruning(self, query):
        block_size = self.block_size

        pruning = []
        R = []
        preproc_time = []
        probability = []
        for i, data in enumerate(query):
            if data is None:
                continue
            rs, r, preproc_t, probability_ = data
            gh_radius = gaussian_heuristic(rs)
            R.append([x/gh_radius for x in rs])
            preproc_time.append(preproc_t)
            probability.append(probability_)

        preproc_time = sum(preproc_time)/len(preproc_time)
        overhead = nodes_per_sec(block_size) * preproc_time
        probability = sum(probability)/len(probability)

        for i in range(-PruningStrategizer.GH_FACTORS_STEPS, PruningStrategizer.GH_FACTORS_STEPS+1):
            radius = gh_margin(block_size) ** (1. * i / PruningStrategizer.GH_FACTORS_STEPS)
            pruning_ = prune(radius, overhead, min(1.001*probability, 0.999), R,
                descent_method=self.pruner_method, precision=self.pruner_precision)
            pruning.append(pruning_)
        return tuple(pruning)


class CopyStrategizer(EmptyStrategizer):
    """
    Strategize as suggested by strategy.

    """
    name = "CopyStrategy"

    def preproc(self, query):
        """No preprocessing.

        :param query: empty

        """
        return self.strategies[self.block_size].preprocessing

    def pruning(self, query):
        """

        :param query: tuple of (r_ii, radius, preprocessing time)

        """
        return self.strategies[self.block_size].pruning_coefficients


def CopyStrategizerFactory(strategies, name="CopyStrategizer"):
    """
    Create CopyStrategizer from strategy list.
    """
    return type("CopyStrategizer", (CopyStrategizer,),
                {"strategies": strategies, "name": name})

