# -*- coding: utf-8 -*-
u"""
Find BKZ reduction strategies using timing experiments.

.. moduleauthor:: Martin R.  Albrecht <fplll-devel@googlegroups.com>
.. moduleauthor:: Léo Ducas  <fplll-devel@googlegroups.com>
.. moduleauthor:: Marc Stevens  <fplll-devel@googlegroups.com>

"""

# We use multiprocessing to parallelize

from __future__ import absolute_import
from multiprocessing import Queue, Pipe, Process, active_children

from fpylll import BKZ, IntegerMatrix
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from fpylll.fplll.bkz_param import Strategy, dump_strategies_json

from strategizer.bkz import CallbackBKZ
from strategizer.bkz import CallbackBKZParam as Param
from strategizer.config import logging, git_revision
from strategizer.util import chunk_iterator
from strategizer.strategizers import PruningStrategizer,\
    OneTourPreprocStrategizerFactory, ProgressivePreprocStrategizerFactory
logger = logging.getLogger(__name__)


def find_best(state, fudge=1.01):
    """
    Given an ordered tuple of tuples, return the minimal one, where
    minimal is determined by first entry.

    :param state:
    :param fudge:


    .. note :: The fudge factor means that we have a bias towards earlier entries.
    """
    best = state[0]
    for s in state:
        if best[0] > fudge*s[0]:
            best = s
    return best


def worker_process(A, params, queue):
    """
    This function is called to collect statistics.

    :param A: basis
    :param params: BKZ parameters
    :param queue: queue used for communication

    """
    bkz = CallbackBKZ(A)
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    with tracer.context(("tour", 0)):
        # TODO interacts badly with initial size reduction
        # with tracer.context("preprocessing"):
        #     # HACK to get preproc time
        #     # bkz.randomize_block(1, params.block_size, tracer, density=params.rerandomization_density)
        #     pass
        bkz.svp_reduction(0, params.block_size, params, tracer)

    tracer.exit()
    # close connection
    params.strategies[params.block_size].connection.send(None)
    queue.put(tracer.trace)


def callback_roundtrip(alive, k, connections, data):
    """
    Send ``data`` on ``connections`` for processes ids in ``alive``, ``k`` at a time.

    :param alive:
    :param k:
    :param connections:
    :param data:
    """
    callback = [None]*len(connections)

    for chunk in chunk_iterator(alive, k):
        for i in chunk:
            connections[i].send(data)

        for i in chunk:
            try:
                callback[i] = connections[i].recv()
            except EOFError:
                callback[i] = None
                connections[i].close()

    return callback


def discover_strategy(block_size, Strategizer, strategies,
                      pruner_method="hybrid",
                      nthreads=1, nsamples=50):
    """Discover a strategy using ``Strategizer``

    :param block_size: block size to try
    :param Strategizer: strategizer to use
    :param strategies: strategies for smaller block sizes
    :param nthreads: number of threads to run
    :param nsamples: number of lattice bases to consider
    :param subprocess:

    """
    connections = []
    processes = []
    k = nthreads
    m = nsamples

    strategizer = Strategizer(block_size, pruner_method=pruner_method)

    # everybody is alive in the beginning
    alive = range(m)

    return_queue = Queue()

    for i in range(m):
        manager, worker = Pipe()
        connections.append((manager, worker))
        A = IntegerMatrix.random(block_size, "qary", bits=30, k=block_size//1, int_type="long")
        strategies_ = list(strategies)
        strategies_.append(Strategizer.Strategy(block_size, worker))

        # note: success probability, rerandomisation density etc. can be adapted here
        param = Param(block_size=block_size, strategies=strategies_, flags=BKZ.VERBOSE)

        process = Process(target=worker_process, args=(A, param, return_queue))
        processes.append(process)

    callback = [None]*m
    for chunk in chunk_iterator(alive, k):
        for i in chunk:
            process = processes[i]
            process.start()
            manager, worker = connections[i]
            worker.close()
            connections[i] = manager

        # wait for `k` responses
        for i in chunk:
            callback[i] = connections[i].recv()

    assert all(callback)  # everybody wants preprocessing parameters

    preproc_params = strategizer(callback)

    callback = callback_roundtrip(alive, k, connections, preproc_params)
    assert all(callback)  # everybody wants pruning parameters

    pruning_params = strategizer(callback)

    callback = callback_roundtrip(alive, k, connections, pruning_params)
    assert not any(callback)  # no more questions

    strategy = Strategy(block_size=block_size,
                        preprocessing_block_sizes=preproc_params,
                        pruning_parameters=pruning_params)

    active_children()

    stats = []
    for i in range(m):
        stats.append(return_queue.get())

    return strategy, tuple(stats), tuple(strategizer.queries)


def strategize(max_block_size,
               existing_strategies=None,
               min_block_size=3,
               nthreads=1, nsamples=50,
               pruner_method="hybrid",
               StrategizerFactory=ProgressivePreprocStrategizerFactory,
               dump_filename=None):
    """
    *one* preprocessing block size + pruning.

    :param max_block_size: maximum block size to consider
    :param strategizers: strategizers to use
    :param existing_strategies: extend these previously computed strategies
    :param min_block_size: start at this block size
    :param nthreads: use this many threads
    :param nsamples: start using this many samples
    :param dump_filename: write strategies to this filename

    """
    if dump_filename is None:
        dump_filename = "default-strategies-%s.json"%git_revision

    if existing_strategies is not None:
        strategies = existing_strategies
        times = [None]*len(strategies)
    else:
        strategies = []
        times = []

    for i in range(len(strategies), min_block_size):
        strategies.append(Strategy(i, [], []))
        times.append(None)

    strategizer = PruningStrategizer

    for block_size in range(min_block_size, max_block_size+1):
        logger.info("= block size: %3d, samples: %3d =", block_size, nsamples)

        state = []

        try:
            p = max(strategies[-1].preprocessing_block_sizes[-1] - 4, 0)
        except (IndexError,):
            p = 0

        prev_best_total_time = None
        while p < block_size:
            if p >= 4:
                strategizer_p = type("PreprocStrategizer-%d"%p,
                                     (strategizer, StrategizerFactory(p)), {})
            else:
                strategizer_p = strategizer

            strategy, stats, queries = discover_strategy(block_size,
                                                         strategizer_p,
                                                         strategies,
                                                         nthreads=nthreads,
                                                         nsamples=nsamples,
                                                         pruner_method=pruner_method,
                                                         )

            stats = [stat for stat in stats if stat is not None]

            total_time = sorted([float(stat.data["cputime"]) for stat in stats])
            svp_time = sorted([float(stat.find("enumeration").data["cputime"]) for stat in stats])
            preproc_time = sorted([float(stat.find("preprocessing").data["cputime"]) for stat in stats])

            total_time = total_time[nsamples//2]
            svp_time = svp_time[nsamples//2]
            preproc_time = preproc_time[nsamples//2]

            state.append((total_time, strategy, stats, strategizer, queries))
            logger.info("%10.6fs, %10.6fs, %10.6fs, %s", total_time, preproc_time, svp_time, strategy)

            if prev_best_total_time and 1.3*prev_best_total_time < total_time:
                break
            p += 2
            if not prev_best_total_time or prev_best_total_time > total_time:
                prev_best_total_time = total_time

        best = find_best(state)
        total_time, strategy, stats, strategizer, queries = best

        strategies.append(strategy)
        dump_strategies_json(dump_filename, strategies)
        times.append((total_time, stats, queries))

        logger.info("")
        logger.info("%10.6fs, %s", total_time, strategy)
        logger.info("")

        if total_time > 0.1 and nsamples > 2*max(32, nthreads):
            nsamples /= 2

    return strategies, times


StrategizerFactoryDictionnary = {
    "ProgressivePreproc": ProgressivePreprocStrategizerFactory,
    "OneTourPreproc": OneTourPreprocStrategizerFactory}

if __name__ == '__main__':
    import argparse
    import logging
    import os

    parser = argparse.ArgumentParser(description='Preprocessing Search')
    parser.add_argument('-t', '--threads', help='number of threads to use', type=int, default=1)
    parser.add_argument('-s', '--samples', help='number of samples to try', type=int, default=16)
    parser.add_argument('-l', '--min-block-size', help='minimal block size to consider', type=int, default=3)
    parser.add_argument('-u', '--max-block-size', help='minimal block size to consider', type=int, default=50)
    parser.add_argument('-f', '--filename', help='json file to store strategies to', type=str, default=None)
    parser.add_argument('-m', '--method', help='descent method for the pruner {gradient,nm,hybrid}',
                        type=str, default="hybrid")
    parser.add_argument('-S', '--strategizer', help='Strategizer : {ProgressivePreproc,OneTourPreproc}',
                        type=str, default="OneTourPreproc")

    args = parser.parse_args()

    log_name = os.path.join("default-strategies-%s.log"%(git_revision))

    if args.filename:
        if not args.filename.endswith(".json"):
            raise ValueError("filename should be a json file")
        log_name = args.filename.replace(".json", ".log")

    extra = logging.FileHandler(log_name)
    extra.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s: %(message)s',)
    extra.setFormatter(formatter)
    logging.getLogger('').addHandler(extra)

    strategize(nthreads=args.threads, nsamples=args.samples,
               min_block_size=args.min_block_size,
               max_block_size=args.max_block_size,
               pruner_method=args.method,
               StrategizerFactory=StrategizerFactoryDictionnary[args.strategizer],
               dump_filename=args.filename)
