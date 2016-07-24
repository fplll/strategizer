# -*- coding: utf-8 -*-
def chunk_iterator(lst, step):
    """Return up to ``step`` entries from ``lst`` each time this function is called.

    :param lst: a list
    :param step: number of elements to return

    """
    for i in xrange(0, len(lst), step):
        yield tuple(lst[j] for j in range(i, min(i+step, len(lst))))
