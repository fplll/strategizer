# -*- coding: utf-8 -*-
"""
Compare strategies
"""


from __future__ import absolute_import
from collections import OrderedDict
from fpylll import BKZ, IntegerMatrix
from fpylll.fplll.bkz_param import Strategy, dump_strategies_json
from fpylll.algorithms.bkz_stats import BKZTreeTracer
from multiprocessing import Queue, Process
from strategizer.bkz import CallbackBKZ
from strategizer.bkz import CallbackBKZParam as Param
from strategizer.config import logging
from strategizer.util import chunk_iterator
import time

logger = logging.getLogger(__name__)


def svp_time(A, params, return_queue=None):
    """Run SVP reduction of AutoBKZ on ``A`` using ``params``.

    :param A: a matrix
    :param params: AutoBKZ parameters
    :param queue: if not ``None``, the result is put on this queue.

    """
    # HACK to count LLL time
    t = time.time()
    bkz = CallbackBKZ(A)
    t = time.time() - t
    tracer = BKZTreeTracer(bkz, start_clocks=True)

    with tracer.context("tour"):
        bkz.svp_reduction(0, params.block_size, params, tracer)

    tracer.exit()

    tracer.trace.data["|A_0|"] = A[0].norm()

    if return_queue:
        return_queue.put(tracer.trace)
    else:
        return tracer.trace


def compare_strategies(strategies_list, nthreads=1, nsamples=50,
                       min_block_size=3, max_block_size=None):

    """Run ``m`` experiments using ``nthreads`` to time one SVP
    reduction for each strategy in ``strategies``.

    :param strategies_list: a list of lists of strategies
    :param nthreads: number of threads
    :param m: number of experiments to run, as the block size
              increases, the number of experiments is automatically
              reduced to ≥ ``max(10,nthreads)``
    :param min_block_size: ignore block sizes smaller than this
    :param max_block_size: ignore block sizes bigger than this
    """
    results = OrderedDict()

    if max_block_size is None:
        max_block_size = min_block_size
        for strategies in strategies_list:
            for strategy in strategies:
                if strategy.block_size > max_block_size:
                    max_block_size = strategy.block_size

    S = dict([(bs, []) for bs in range(min_block_size, max_block_size+1)])
    for strategies in strategies_list:
        for strategy in strategies:
            if strategy.block_size not in S:
                logger.warning("ignoring block_size: %3d of %s", strategy.block_size, strategy)
                continue
            S[strategy.block_size].append(strategies)

    results = [[] for bs in range(max_block_size+1)]

    for block_size in range(min_block_size, max_block_size+1):
        logger.info("= block size: %3d, m: %3d =", block_size, nsamples)
        for strategies in S[block_size]:

            return_queue = Queue()
            result = OrderedDict([("strategy", strategies[block_size]),
                                  ("total time", None)])

            stats = []
            # 2. run `k` processes in parallel until first callback
            for chunk in chunk_iterator(range(nsamples), nthreads):
                processes = []
                for i in chunk:
                    A = IntegerMatrix.random(block_size, "qary", bits=30, k=block_size//2, int_type="long")
                    param = Param(block_size=block_size,
                                  strategies=strategies,
                                  flags=BKZ.VERBOSE)
                    process = Process(target=svp_time, args=(A, param, return_queue))
                    processes.append(process)
                    process.start()

                for process in processes:
                    process.join()
                    stats.append(return_queue.get())

            total_time = sum([float(stat.data["cputime"]) for stat in stats])/nsamples
            length    = sum([stat.data["|A_0|"] for stat in stats])/nsamples
            logger.info("%10.6fs, %s, %.1f"%(total_time, strategies[block_size], length))

            result["total time"] = total_time
            result["length"] = length
            result["stats"] = stats

            results[block_size].append(result)

        if results[block_size][0]["total time"] > 1.0 and nsamples > 2*max(32, nthreads):
            nsamples /= 2

    return results


if __name__ == '__main__':
    import argparse
    import json
    import os
    from fpylll.fplll.bkz_param import load_strategies_json
    parser = argparse.ArgumentParser(description='Benchmarketing')
    parser.add_argument('-t', '--threads', help='number of threads to use', type=int, default=1)
    parser.add_argument('-s', '--samples', help='number of samples to try', type=int, default=10)
    parser.add_argument('-l', '--min-block-size', help='minimal block size to consider', type=int, default=3)
    parser.add_argument('-u', '--max-block-size', help='maximal block size to consider', type=int, default=None)
    parser.add_argument('strategies', help='jsons file to load strategies from', type=str, nargs='*')

    args = parser.parse_args()

    name = ",".join([os.path.basename(strategy).replace(".json", "") for strategy in args.strategies])
    log_name = "compare-%s.log"%name
    extra = logging.FileHandler(log_name)
    extra.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name)s: %(message)s',)
    extra.setFormatter(formatter)
    logging.getLogger('').addHandler(extra)

    strategiess = []
    for strategies in args.strategies:
        strategiess.append(load_strategies_json(strategies))

    results = compare_strategies(strategiess, nthreads=args.threads, nsamples=args.samples,
                                 min_block_size=args.min_block_size, max_block_size=args.max_block_size)
    json_dict = OrderedDict()

    best = [Strategy(bs) for bs in range(args.max_block_size+1)]

    for result in results:
        if not result:
            continue
        json_dict[result[0]["strategy"].block_size] = []

        min_t = None

        for entry in result:
            d = OrderedDict()
            d["name"] = str(entry["strategy"])
            d["total time"] = entry["total time"]
            d["length"] = entry["length"]
            json_dict[result[0]["strategy"].block_size].append(d)

            if min_t is None or min_t > entry["total time"]:
                best[result[0]["strategy"].block_size] = entry["strategy"]
                min_t = entry["total time"]

    json_name = "compare-%s.json"%(name)
    json.dump(json_dict, open(json_name, "w"), indent=4, sort_keys=False)
    best_name = "compare-best-%s.json"%(name)
    dump_strategies_json(best_name, best)
